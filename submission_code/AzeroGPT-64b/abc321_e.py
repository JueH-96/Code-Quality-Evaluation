# T test cases
for _ in range(int(input())):
    N, X, K = map(int, input().split())

    # For K=0, the answer is vertex X itself if it exists (N >= X)
    if K == 0:
        print(1 if N >= X else 0)
        continue

    # Calculate the highest power of 2 <= X
    X_power_of_2 = 1
    while X_power_of_2 * 2 <= X:
        X_power_of_2 *= 2

    # Calculate the range of vertices after K steps
    min_vertex = max(X_power_of_2 // pow(2, K), 1)
    max_vertex = min(N, X_power_of_2 * 2 - 1)

    print(max(0, min(max_vertex, X_power_of_2 * pow(2, K - 1)) - max_vertex + 1))