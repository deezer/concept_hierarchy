"""
Hierarchy extraction
"""

import heapq as hq
from collections import defaultdict
from time import time

import networkx as nx
import numpy as np
from tqdm import tqdm


def gen_sim_graph(S, threshold=0.5):
    similarity = np.copy(S)  # S
    np.fill_diagonal(similarity, np.min(similarity) - 0.1)
    adjacency = similarity > threshold  # A

    return similarity, adjacency


"""
Implements Heymann and Garcia-Molina (2006)

... and extended to handle top K.
"""


def heymann(
    graph,
    similarity,
    centrality_vec={},
    min_sim=0.2,
    parents_per_node=1,
    additional_threshold=0.2,
    max_children_per_node=20,
):
    """
    Transforms a graph into a tree hierarchy.
    if centrality is provided, graph will not be used.

    graph: nx graph to use to compute the centrality vec
    similarity: similarity matrix
    std_times_threshold : heymann min threshold to add node not to root
    parents_per_node : add several #link to node, to control sparsity
    additional_threshold : avoids bad links for multiple parents
    max_children_per_node : avoids big cluster of nodes

    return matrix of [parent, children]
    """
    if len(centrality_vec) == 0:
        print("computing centrality...")
        start_time = time()
        centrality_vec = nx.centrality.betweenness_centrality(graph)
        print("took", time() - start_time, "seconds")
    sorted_nodes = sorted(centrality_vec.items(), key=lambda el: el[1])[::-1]

    heymann_graph = np.zeros((similarity.shape[1] + 1, similarity.shape[1]))  # + root
    count_nodes = defaultdict(int)
    count_nodes["<root>"] = -1e9

    for i, (node, v) in tqdm(enumerate(sorted_nodes)):
        if i == 0:  # solo node
            heymann_graph[-1, node] = True
            continue

        max_heap = [(float("inf"), "<root>")]
        for node_idx, w in sorted_nodes[:i]:  # already added
            if count_nodes[node_idx] >= max_children_per_node:
                continue  # max node by node, find something else
            sim = similarity[node_idx, node]  # important not to use the .T of this
            hq.heappush(max_heap, (-sim, node_idx))

        parent_candidates = hq.nsmallest(
            parents_per_node, max_heap
        )  # extract top similarities
        max_negsim, max_idx = max_heap[0]
        for negsim, idx in parent_candidates:
            if (-negsim > min_sim) and (-negsim > -(max_negsim + additional_threshold)):
                heymann_graph[idx, node] = True
                count_nodes[idx] += 1
            else:
                heymann_graph[-1, node] = True

    print(
        "heymann sparsity : {} (#:{})".format(
            np.sum(heymann_graph) / heymann_graph.shape[0] ** 2, np.sum(heymann_graph)
        )
    )

    return heymann_graph


# utility: reduced graph for sub-evaluations


def regen_subgraph(idx, S, threshold=0.5):
    """
    Regen graph on only the available subet of nodes
    """
    S_red = S[idx][:, idx]
    return gen_sim_graph(S_red, threshold)


def regen_subhierarchy(idx, S, names, centrality_vec):
    """
    Regen a hierarchy from a subset of nodes
    """
    names_red = np.array(names)[idx]
    centrality_red = {}
    for i, name in enumerate(names_red):
        old_idx = names.index(name)
        centrality_red[i] = centrality_vec[old_idx]

    similarity, _ = regen_subgraph(idx, S)
    heymann_graph = heymann({}, similarity, centrality_red)

    return similarity, heymann_graph


def cosim(A, B):
    """<A,B>/|A||B|"""
    na = np.sqrt(A @ A)
    nb = np.sqrt(B @ B)
    if na * nb == 0:
        return 0.0
    return (A @ B) / (na * nb)


def nn_dist(X):
    norm = np.sum(np.square(X), -1, keepdims=True)
    return norm + norm.T - 2 * np.dot(X, X.T)


def gen_opt_graph(names, emb, use_cosim=True, threshold=0.1, gen_graph=False):
    """
    I don't know what to do exactly as it's not build on CAVs. I will do very generically:
    - threshold sim to add base graph
    - add top1 by default to avoid unconnected nodes
    """
    E = np.stack([emb[pid] for pid in names], 0)
    if use_cosim:
        E = E / np.sqrt(np.sum(np.square(E), 1, keepdims=True))
        similarity = E @ E.T
        np.fill_diagonal(similarity, -1)  # no self-loop
    else:
        similarity = -nn_dist(E)
        np.fill_diagonal(similarity, np.min(similarity) - 0.1)

    adjacency = np.zeros_like(similarity)
    adjacency = similarity == np.max(similarity, axis=1, keepdims=True)  # min top con
    adjacency = adjacency | (similarity > threshold)  # and add thresholded
    print(
        "sim graph sparsity : {} (#:{})".format(
            np.sum(adjacency) / adjacency.shape[0] ** 2, np.sum(adjacency)
        )
    )

    if not gen_graph:
        return adjacency, similarity, None

    graph = nx.Graph()
    for v in range(adjacency.shape[1]):
        graph.add_node(v)
    for v in range(adjacency.shape[1]):
        for w in range(adjacency.shape[1]):
            if v != w and adjacency[v, w]:
                graph.add_edge(v, w)
    return adjacency, similarity, graph


# eval graph with embs


def metric_dist(names, emb, graph, use_cosim=True):
    nei_dist = []
    for node, graph_parent in zip(names, graph.T):
        parents = np.where(graph_parent == 1)[0]
        if parents.shape[0] == 0:
            continue
        for parent in parents:
            if use_cosim:
                diff_sim = cosim(emb[node], emb[names[parent]])
            else:
                diff_sim = np.sqrt(np.sum(np.square(emb[node] - emb[names[parent]])))
            nei_dist.append(diff_sim)
    conf_coef = 1.96 / np.sqrt(len(nei_dist))
    return np.mean(nei_dist), np.std(nei_dist), conf_coef * np.std(nei_dist)


def eval_unsupervised_audio(S, names, hierarchy=True, centrality_vec={}, baseline=-1):
    """
    Eval on a set of metrics and with different possible baselines.
    """
    latex_script = ""

    def eval_sub(subnames, subemb, use_cosim=True, emb_name="subgraph"):
        """I input subnames instead of computing it to keep track of the
        exact ordered list."""
        idx = [names.index(el) for el in subnames if el in total_inter]
        filtered_subnames = [el for el in subnames if el in total_inter]
        if hierarchy:
            similarity, subgraph = regen_subhierarchy(idx, S, names, centrality_vec)
            subgraph = subgraph[:-1]
        else:
            if baseline == 1:
                similarity, subgraph = regen_subgraph(idx, S, 0.66)
            elif baseline == 0:  # random graph
                similarity, _ = regen_subgraph(
                    idx, S, 0.1
                )  # just a way to have similarity
                rand_graph = np.random.uniform(0, 1, (len(idx), len(idx)))
                subgraph = rand_graph == np.max(rand_graph, axis=1, keepdims=True)
            elif baseline == 3:  # random graph
                similarity, _ = regen_subgraph(
                    idx, S, 0.1
                )  # just a way to have similarity
                rand_graph = np.random.uniform(0, 1, (len(idx), len(idx)))
                subgraph = rand_graph > 0.9
            else:
                similarity, subgraph = regen_subgraph(idx, S)

            if baseline == 2:
                subgraph = similarity == np.max(similarity, 1, keepdims=True)

        mean_dist, std_dist, conf_dist = metric_dist(
            filtered_subnames, subemb, subgraph, use_cosim
        )
        print(
            "{}: cosim neighbors {:.4f} ± {:.4f} (std: {:.4f})".format(
                emb_name, mean_dist, conf_dist, std_dist
            )
        )
        return "\n & {:.3f} $\\pm$ {:.3f}".format(mean_dist, conf_dist)

    latex_script += eval_sub(names, emb_music, use_cosim=False, emb_name="Music")
    latex_script += eval_sub(names_usageemb, emb_usg, use_cosim=False, emb_name="Usage")
    latex_script += eval_sub(names_llm, emb_llm, emb_name="BERT")
    latex_script += eval_sub(names_w2v_mwon, emb_w2v_mwon, emb_name="W2V Won")
    latex_script += eval_sub(names_w2v_sdoh, emb_w2v_sdoh, emb_name="W2V Doh")

    return latex_script


def eval_opt(inclusion, centrality, names):
    """Same but without baseline, more straightforward for OPT graphs."""
    latex_script = ""

    def regen_and_eval(subemb, use_cosim=True, emb_name="subgraph"):
        subnames = list(set(subemb.keys()) & set(names))
        idx = [names.index(el) for el in subnames if el in total_inter]
        filtered_subnames = [el for el in subnames if el in total_inter]
        subsimilarity = inclusion[idx][:, idx]
        new_centrality = {}
        for i, j in enumerate(idx):
            new_centrality[i] = centrality[j]
        if np.mean(inclusion) > -0.5:
            subgraph = heymann({}, subsimilarity, new_centrality)
        else:
            subgraph = heymann({}, subsimilarity, new_centrality, min_sim=-3.0)
        subgraph = subgraph[:-1]  # delete <root>

        mean_dist, std_dist, conf_dist = metric_dist(
            filtered_subnames, subemb, subgraph, use_cosim
        )
        print(
            "{}: cosim neighbors {:.4f} ± {:.4f} (std: {:.4f})".format(
                emb_name, mean_dist, conf_dist, std_dist
            )
        )
        return "\n & {:.3f} $\\pm$ {:.3f}".format(mean_dist, conf_dist)

    latex_script += regen_and_eval(emb_music, use_cosim=False, emb_name="Music")
    latex_script += regen_and_eval(emb_usg, use_cosim=False, emb_name="Usage")
    latex_script += regen_and_eval(emb_llm, emb_name="BERT")
    latex_script += regen_and_eval(emb_w2v_mwon, emb_name="W2V Won")
    latex_script += regen_and_eval(emb_w2v_sdoh, emb_name="W2V Doh")

    return latex_script


if __name__ == "__main__":
    playlists = np.load("../data/deezer_playlists.npy", allow_pickle=True).item()

    # stats_deezer, stats_genre_apm, stats_mood_apm
    stat_dict = np.load("../results/stats_deezer.npy", allow_pickle=True).item()

    training_names = np.load("../deezer_cav.npy", allow_pickle=True)[2]
    # training_names = list(playlists.keys())
    names = []
    last_pid = -1
    filtered_idx = []

    # ------------------------------------------------------------------------------------------------------------------
    # quartile graph
    for k in stat_dict.keys():
        pid = int(k.split("_")[0])
        if pid == last_pid:
            continue
        last_pid = pid
        names.append(pid)
        filtered_idx.append(training_names.index(pid))

    studied_pooling_layer = 2
    act = "sig"  # '' or 'sig'

    S = np.zeros((len(names), len(names)))
    for i, k in enumerate(names):
        layer_tag = f"{k}_{studied_pooling_layer}"
        S[i, :] = (
            stat_dict[layer_tag][f"e{act}x"][filtered_idx] / stat_dict[layer_tag]["n"]
        )

    similarity, adjacency = gen_sim_graph(S, 0.5)
    print(
        "sparsity : {} (#:{})".format(
            np.sum(adjacency) / adjacency.shape[0] ** 2, np.sum(adjacency)
        )
    )

    ## create undirected graph

    graph = nx.Graph()
    for v in range(adjacency.shape[1]):
        graph.add_node(v)
    for v in range(adjacency.shape[1]):
        for w in range(adjacency.shape[1]):
            if v != w and adjacency[v, w]:
                graph.add_edge(v, w)

    ## precompute and store centrality  (~20min for Deezer)

    # print("computing centrality...")
    # start_time = time()
    # centrality_vec = nx.centrality.betweenness_centrality(graph)
    # print("took", time() - start_time, "seconds")
    # np.save("../results/centrality_deezer.npy", centrality_vec)
    # del centrality_vec

    # ------------------------------------------------------------------------------------------------------------------
    # Hierarchy Heymann

    # heymann_graph = heymann(graph, similarity, parents_per_node = 1 )
    deezer_centrality_vec = np.load(
        "../results/centrality_deezer.npy", allow_pickle=True
    ).item()
    heymann_graph = heymann({}, similarity, deezer_centrality_vec, parents_per_node=1)

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation

    # -- LLM -> warning, LLM has more names than the cav matrix
    emb_llm = np.load("../results/emb_bert.npy", allow_pickle=True).item()
    names_llm = list(set(emb_llm.keys()) & set(names))
    llm_intersect, llm_similarity, sim_llm_graph = gen_opt_graph(
        names_llm, emb_llm, threshold=0.43
    )
    llm_centrality_vec = nx.centrality.betweenness_centrality(sim_llm_graph)

    # USG
    emb_usg = np.load(
        "../results/emb_collaborative_filtering.npy", allow_pickle=True
    ).item()
    names_usageemb = list(set(emb_usg.keys()) & set(names))
    usg_intersect, usg_similarity, sim_usg_graph = gen_opt_graph(
        names_usageemb, emb_usg, use_cosim=False, threshold=-0.8, gen_graph=True
    )
    usg_centrality_vec = nx.centrality.betweenness_centrality(sim_usg_graph)

    # W2V S.DOH
    emb_w2v_sdoh = np.load("../results/emb_sdoh_w2v.npy", allow_pickle=True).item()
    names_w2v_sdoh = list(set(names) & set(emb_w2v_sdoh.keys()))
    sdoh_intersect, sdoh_similarity, sim_sdoh_graph = gen_opt_graph(
        names_w2v_sdoh, emb_w2v_sdoh, threshold=0.68
    )
    sdoh_centrality_vec = nx.centrality.betweenness_centrality(sim_sdoh_graph)

    # W2V M.WON
    emb_w2v_mwon = np.load("../results/emb_mwon_w2v.npy", allow_pickle=True).item()
    names_w2v_mwon = list(set(names) & set(emb_w2v_mwon.keys()))
    mwon_intersect, mwon_similarity, sim_mwon_graph = gen_opt_graph(
        names_w2v_mwon, emb_w2v_mwon, threshold=0.38
    )
    mwon_centrality_vec = nx.centrality.betweenness_centrality(sim_mwon_graph)

    # Audio
    weights_np, biases_np, names_cav = np.load(
        "weights/learned_deezer_cav.npy", allow_pickle=True
    )
    emb_music_tensor = np.concatenate((weights_np[2], biases_np[2]), 0).T
    emb_music = {}
    for emb, name in zip(emb_music_tensor, names_cav):
        emb_music[name] = emb

    total_inter = list(
        set(names_llm) & set(names_w2v_mwon) & set(names_w2v_sdoh) & set(names_usageemb)
    )

    # ------------------------------------------------------------------------------------------------------------------
    # eval graph with embs

    latex_script = ""
    latex_script += (
        "$H$"
        + eval_unsupervised_audio(S, names, centrality_vec=deezer_centrality_vec)
        + " \\\\ \n"
    )

    latex_script += "\n\\hline \n"
    latex_script += (
        "$H_\\textrm{CF}$"
        + eval_opt(usg_similarity, usg_centrality_vec, names_usageemb)
        + " \\\\ \n"
    )
    latex_script += (
        "$H_\\textrm{BERT}$"
        + eval_opt(llm_similarity, llm_centrality_vec, names_llm)
        + " \\\\ \n"
    )
    latex_script += (
        "$H_\\textrm{W2V-1}$"
        + eval_opt(mwon_similarity, mwon_centrality_vec, names_w2v_mwon)
        + " \\\\ \n"
    )
    latex_script += (
        "$H_\\textrm{W2V-2}$"
        + eval_opt(sdoh_similarity, sdoh_centrality_vec, names_w2v_sdoh)
        + " \\\\"
    )

    latex_script += "\n\\hline \n"
    latex_script += (
        "Random (Top)"
        + eval_unsupervised_audio(
            S, names, hierarchy=False, centrality_vec=deezer_centrality_vec, baseline=0
        )
        + " \\\\ \n"
    )
    latex_script += (
        "Random (Sim)"
        + eval_unsupervised_audio(
            S, names, hierarchy=False, centrality_vec=deezer_centrality_vec, baseline=3
        )
        + " \\\\ \n"
    )
    latex_script += (
        "Similarity"
        + eval_unsupervised_audio(
            S, names, hierarchy=False, centrality_vec=deezer_centrality_vec
        )
        + " \\\\ \n"
    )
    latex_script += (
        "Matching"
        + eval_unsupervised_audio(
            S, names, hierarchy=False, centrality_vec=deezer_centrality_vec, baseline=1
        )
        + " \\\\ \n"
    )
    latex_script += (
        "Top-1"
        + eval_unsupervised_audio(
            S, names, hierarchy=False, centrality_vec=deezer_centrality_vec, baseline=2
        )
        + " \\\\ \n"
    )
    latex_script += "\n\\hline \n"

    print(latex_script)
