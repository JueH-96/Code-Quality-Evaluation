from collections import defaultdict
from heapq import heappush, heappop

def main():
    N, M = map(int, input().split())
    edges = defaultdict(set)
    for _ in range(M):
        u, v = map(int, input().split())
        edges[u].add(v)
        edges[v].add(u)
    weights = list(map(int, input().split()))
    pieces = list(map(int, input().split()))

    # Calculate the maximum number of pieces that can be placed on each vertex
    max_pieces = [0] * (N + 1)
    for v in range(1, N + 1):
        max_pieces[v] = pieces[v - 1]
        for u in edges[v]:
            if weights[u - 1] < weights[v - 1]:
                max_pieces[v] += max_pieces[u]

    # Use a priority queue to process vertices in decreasing order of weight
    pq = []
    for v in range(1, N + 1):
        heappush(pq, (-weights[v - 1], v))
    answer = 0

    while pq:
        _, v = heappop(pq)
        if max_pieces[v] > 0:
            answer += 1
            max_pieces[v] -= 1
            for u in edges[v]:
                if weights[u - 1] < weights[v - 1]:
                    max_pieces[u] -= 1

    print(answer)

if __name__ == "__main__":
    main()