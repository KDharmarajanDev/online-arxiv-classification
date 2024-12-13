from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    NUM_VISUALIZED_NODES = 500

    dataset = PygNodePropPredDataset(name = "ogbn-arxiv", root = 'dataset/')
    graph = dataset[0]

    node_indices = np.random.choice(graph.num_nodes, 10000, replace=False)
    visualized_embeddings = graph.x[node_indices]
    visualized_labels = graph.y[node_indices].squeeze()
    tsne = TSNE(n_components=2, random_state=0)

    subject_areas_per_label_idx = [
        "cs.NA", "cs.MM", "cs.LO", "cs.CY", "cs.CR", "cs.DC", "cs.HC", "cs.CE",
        "cs.NI", "cs.CC", "cs.AI", "cs.MA", "cs.GL", "cs.NE", "cs.SC", "cs.AR",
        "cs.CV", "cs.GR", "cs.ET", "cs.SY", "cs.CG", "cs.OH", "cs.PL", "cs.SE",
        "cs.LG", "cs.SD", "cs.SI", "cs.RO", "cs.IT", "cs.PF", "cs.CL", "cs.IR",
        "cs.MS", "cs.FL", "cs.DS", "cs.OS", "cs.GT", "cs.DB", "cs.DL", "cs.DM"
    ]

    visualized_categories = [
        "cs.CV",
        "cs.LG",
        "cs.IT",
        "cs.CL",
        "cs.AI",
        "cs.DS",
        "cs.NI",
        "cs.CR",
        "cs.DC",
        "cs.LO"
    ]

    category_abbreviation_to_full = {
        "cs.AI": "Artificial Intelligence",
        "cs.AR": "Hardware Architecture",
        "cs.CC": "Computational Complexity",
        "cs.CE": "Computational Engineering, Finance, and Science",
        "cs.CG": "Computational Geometry",
        "cs.CL": "Computation and Language",
        "cs.CR": "Cryptography and Security",
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.CY": "Computers and Society",
        "cs.DB": "Databases",
        "cs.DC": "Distributed, Parallel, and Cluster Computing",
        "cs.DL": "Digital Libraries",
        "cs.DM": "Discrete Mathematics",
        "cs.DS": "Data Structures and Algorithms",
        "cs.ET": "Emerging Technologies",
        "cs.FL": "Formal Languages and Automata Theory",
        "cs.GL": "General Literature",
        "cs.GR": "Graphics",
        "cs.GT": "Computer Science and Game Theory",
        "cs.HC": "Human-Computer Interaction",
        "cs.IR": "Information Retrieval",
        "cs.IT": "Information Theory",
        "cs.LG": "Machine Learning",
        "cs.LO": "Logic in Computer Science",
        "cs.MA": "Multiagent Systems",
        "cs.MM": "Multimedia",
        "cs.MS": "Mathematical Software",
        "cs.NA": "Numerical Analysis",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.NI": "Networking and Internet Architecture",
        "cs.OH": "Other Computer Science",
        "cs.OS": "Operating Systems",
        "cs.PF": "Performance",
        "cs.PL": "Programming Languages",
        "cs.RO": "Robotics",
        "cs.SC": "Symbolic Computation",
        "cs.SD": "Sound",
        "cs.SE": "Software Engineering",
        "cs.SI": "Social and Information Networks",
        "cs.SY": "Systems and Control",
    }

    visualized_label_idx = [subject_areas_per_label_idx.index(category) for category in visualized_categories]
    embedding_selection_mask = np.isin(visualized_labels, visualized_label_idx)

    labels = visualized_labels[embedding_selection_mask][:NUM_VISUALIZED_NODES]
    skip_gram_embeddings_2d = tsne.fit_transform(visualized_embeddings[embedding_selection_mask][:NUM_VISUALIZED_NODES])

    visualized_embeddings = np.load("stella_embedings.npy", allow_pickle=True)[node_indices]
    stella_embeddings_2d = tsne.fit_transform(visualized_embeddings[embedding_selection_mask][:NUM_VISUALIZED_NODES])

    embeddings = torch.load("deberta_fixed_newline.pt").x.cpu().numpy()
    embeddings = embeddings[node_indices]
    deberta_embeddings_2d = tsne.fit_transform(embeddings[embedding_selection_mask][:NUM_VISUALIZED_NODES])

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, len(visualized_categories)))

    # Plotting data on each subplot
    for i, visualized_category in enumerate(visualized_categories):
        label_idx = subject_areas_per_label_idx.index(visualized_category)
        
        category_full = category_abbreviation_to_full[visualized_category]
        axs[0].scatter(skip_gram_embeddings_2d[labels == label_idx, 0], skip_gram_embeddings_2d[labels == label_idx, 1], c=colors[i], label=category_full)
        axs[1].scatter(deberta_embeddings_2d[labels == label_idx, 0], deberta_embeddings_2d[labels == label_idx, 1], c=colors[i], label=category_full)
        axs[2].scatter(stella_embeddings_2d[labels == label_idx, 0], stella_embeddings_2d[labels == label_idx, 1], c=colors[i], label=category_full)

    axs[0].set_title('Skip Gram')
    axs[1].set_title('DeBERTa')
    axs[2].set_title('Stella')

    plt.legend(title="Categories", ncol=1, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.suptitle("t-SNE Visualization of Different Embedding Methods", fontsize=20, fontweight='bold')

    # Adjust layout
    plt.tight_layout()
    plt.show()