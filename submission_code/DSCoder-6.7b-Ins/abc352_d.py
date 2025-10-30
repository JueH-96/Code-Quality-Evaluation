def min_index_difference():
    N, K = map(int, input().split())
    P = list(map(int, input().split()))

    # Create a dictionary to store the indices of each element in the permutation
    indices = {P[i]: i+1 for i in range(N)}

    # Sort the indices
    sorted_indices = sorted(indices.values())

    # Find the minimum difference
    min_diff = float('inf')
    for i in range(K-1, N):
        diff = sorted_indices[i] - sorted_indices[0]
        if diff < min_diff:
            min_diff = diff

    return min_diff

print(min_index_difference())