# YOUR CODE HERE
import sys
import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    M = int(input[idx + 1])
    X1 = int(input[idx + 2])
    idx += 3

    trains = []
    for i in range(M):
        A = int(input[idx])
        B = int(input[idx + 1])
        S = int(input[idx + 2])
        T = int(input[idx + 3])
        trains.append((A, B, S, T))
        idx += 4

    # Create a graph where each node is a city and edges are trains
    graph = [[] for _ in range(N + 1)]
    for i, (A, B, S, T) in enumerate(trains):
        graph[A].append((B, S, T, i))

    # Initialize the delay array with 0
    X = [0] * M
    X[0] = X1

    # Priority queue for Dijkstra's algorithm
    pq = []
    heapq.heappush(pq, (X1, 1, 0))  # (current delay, current city, train index)

    while pq:
        current_delay, current_city, train_idx = heapq.heappop(pq)

        for next_city, S, T, next_train_idx in graph[current_city]:
            if next_train_idx == 0:
                continue  # Skip the first train

            required_delay = max(0, T + current_delay - S)
            if required_delay > X[next_train_idx]:
                X[next_train_idx] = required_delay
                heapq.heappush(pq, (required_delay, next_city, next_train_idx))

    # Print the result excluding X1
    print(" ".join(map(str, X[1:])))

if __name__ == "__main__":
    main()