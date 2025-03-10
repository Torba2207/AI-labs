import numpy as np

def initialize_centroids_forgy(data, k):
    # Randomly select k data points as initial centroids (Forgy method)
    #np.random.seed(42)  # For reproducibility
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    return data[indices]

def initialize_centroids_kmeans_pp(data, k):

    centroids = [data[np.random.randint(data.shape[0])]]
    for _ in range(1, k):

        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in data])
        #distances = np.array([sum([np.linalg.norm(x - c) for c in centroids]) for x in data])
        new_centroid_id=np.argmax(distances)
        centroids.append(data[new_centroid_id])

        #for j, p in enumerate(cumprobs):
        #    if rand_value < p:
        #        index = j
        #        break
        #centroids.append(data[index])
    return np.array(centroids)

def assign_to_cluster(data, centroids):
    # Find the closest cluster for each data point
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, assignments):
    # Update centroids based on the assignments
    centroids = np.array([np.mean(data[assignments == i], axis=0) for i in range(np.max(assignments) + 1)])
    return centroids


def mean_intra_distance(data, assignments, centroids):
    return np.sqrt(np.sum((data - centroids[assignments, :])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else:
        centroids = initialize_centroids_forgy(data, num_centroids)


    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)

