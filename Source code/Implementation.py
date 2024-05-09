#Graph Base KNN Extension
import os
import networkx as nx
import time

def text_to_graph(text):
    words = text.split()
    G = nx.Graph()
    for word in set(words):
        G.add_node(word)
    for w1, w2 in zip(words[:-1], words[1:]):
        if not G.has_edge(w1, w2):
            G.add_edge(w1, w2)
    return G

def distMCS(G1, G2):
    mcs_size = compute_mcs_size(G1, G2)
    max_size = max(G1.number_of_nodes() + G1.number_of_edges(), G2.number_of_nodes() + G2.number_of_edges())
    return 1 - mcs_size / max_size

def compute_mcs_size(G1, G2):
    common_nodes = len(set(G1.nodes()) & set(G2.nodes()))
    common_edges = len(set(G1.edges()) & set(G2.edges()))
    return common_nodes + common_edges

def assign_class(filename):
    if 'LH' in filename:
        return 'c1'
    elif 'DS' in filename:
        return 'c2'
    elif 'TR' in filename:
        return 'c3'
    else:
        return None

def find_nearest_training_instance(test_directory, train_directory):
    test_files = os.listdir(test_directory)
    train_files = os.listdir(train_directory)
    nearest_instances = {}
    for test_filename in test_files:
        with open(os.path.join(test_directory, test_filename), 'r', encoding='utf-8') as file:
            test_text = file.read()
        test_graph = text_to_graph(test_text)
        distances = {}
        for train_filename in train_files:
            with open(os.path.join(train_directory, train_filename), 'r', encoding='utf-8') as file:
                train_text = file.read()
            train_graph = text_to_graph(train_text)
            distance = distMCS(test_graph, train_graph)
            distances[train_filename] = distance
        nearest_training_instances = sorted(distances, key=distances.get)[:5]
        class_counts = {'c1': 0, 'c2': 0, 'c3': 0}
        for instance in nearest_training_instances:
            class_label = assign_class(instance)
            if class_label:
                class_counts[class_label] += 1
        predicted_class = max(class_counts, key=class_counts.get)
        nearest_instances[test_filename] = predicted_class
    return nearest_instances

def print_class_label_mappings():
    print("Class c1 corresponds to 'LifeStyle_and_Hobbies'")
    print("Class c2 corresponds to 'Disease_and_Symptoms'")
    print("Class c3 corresponds to 'Travel'")
    print()

def print_table_header():
    print("{:<40} {:<20} {:<20}".format("All Testing Files (names)", "Predicted class", "Actual class"))
    print("-" * 80)

def print_results(nearest_instances, actual_classes):
    for test_file, predicted_class in nearest_instances.items():
        actual_class = actual_classes.get(test_file, "Unknown")
        print("{:<40} {:<20} {:<20}".format(test_file, predicted_class, actual_class))

test_directory = r'E:\G11_(622,641)\Data\Testing_files'
train_directory = r'E:\G11_(622,641)\Data\Training_files'

actual_classes = {'D13_DS_T.txt': 'c2', 'D13_TR_T.txt': 'c3', 'D14_DS_T.txt': 'c2', 
                  'D14_TR_T.txt': 'c3', 'D15_DS_T.txt': 'c2', 'D15_LH_T.txt': 'c1', 
                  'D15_TR_T.txt': 'c3', 'D4_LH_T.txt': 'c1', 'D6_LH_T.txt': 'c1'}

print_class_label_mappings()

start_time = time.time()
nearest_instances = find_nearest_training_instance(test_directory, train_directory)
end_time = time.time()

print_table_header()
print_results(nearest_instances, actual_classes)
print("\nTime of execution:", round(end_time - start_time, 3), "seconds")
