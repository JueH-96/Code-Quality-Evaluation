def find_bridge_index(N, M, X):
    # Calculate the minimum distance to reach each X[i] from the starting point 0
    distance_to_X = [0] * M
    prev = 0
    for i, x in enumerate(X):
        distance_to_X[i] = min(prev + 1, abs(x - prev) + 1)
        prev = x

    # Calculate the minimum distance from the end of the tour to each X[i]
    distance_from_X_M = [0] * M
    prev = 0
    for i in range(M - 1, -1, -1):
        distance_from_X_M[i] = min(prev + 1, abs(X[i] - X[-1]) + 1)
        prev = distance_from_X_M[i]

    # The distance around the archipelago, moving from X[M-1] to X[0] via the remaining bridges
    full_cycle_distance = abs(X[-1] - X[0]) + 1

    min_length = full_cycle_distance + M - 1
    for i in range(len(distance_to_X)):
        current_distance = distance_to_X[i] + distance_from_X_M[i + 1]
        if current_distance < full_cycle_distance and i + 1 < M and X[i] != X[i + 1]:
            min_length = min(min_length, current_distance + M - 2)
        else:
            min_length = min(min_length, current_distance + M - 1)

    return min_length

# Read inputs from stdin
N, M = map(int, input().split())
X = list(map(int, input().split()))

# Solve the problem and print the result to stdout
print(find_bridge_index(N, M, X))