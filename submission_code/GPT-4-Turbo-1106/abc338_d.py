def main():
    N, M = map(int, input().split())
    X = list(map(int, input().split()))

    # Calculate the distances between consecutive islands in the tour
    distances = [0] * (M - 1)
    for i in range(M - 1):
        distances[i] = (X[i + 1] - X[i]) % N
        if distances[i] > N // 2:
            distances[i] = N - distances[i]

    # Calculate the total distance of the tour without closing any bridge
    total_distance = sum(distances)

    # Calculate the prefix sum of distances
    prefix_sum = [0] * M
    for i in range(1, M):
        prefix_sum[i] = prefix_sum[i - 1] + distances[i - 1]

    # Calculate the suffix sum of distances
    suffix_sum = [0] * M
    for i in range(M - 2, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + distances[i]

    # Find the minimum tour length by trying to close each bridge
    min_tour_length = total_distance
    for i in range(M - 1):
        # Calculate the distance if the bridge between X[i] and X[i+1] is closed
        distance_with_closed_bridge = total_distance - distances[i] + min(prefix_sum[i], suffix_sum[i + 1])
        min_tour_length = min(min_tour_length, distance_with_closed_bridge)

    # Output the minimum tour length
    print(min_tour_length)

if __name__ == "__main__":
    main()