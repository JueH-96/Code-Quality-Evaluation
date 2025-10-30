import sys
from collections import deque, defaultdict

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N_1 = int(data[index])
    N_2 = int(data[index + 1])
    M = int(data[index + 2])
    index += 3

    edges = defaultdict(list)

    for _ in range(M):
        a = int(data[index])
        b = int(data[index + 1])
        index += 2
        edges[a].append(b)
        edges[b].append(a)

    def bfs(start, group_size):
        dist = [-1] * (N_1 + N_2 + 1)
        queue = deque([start])
        dist[start] = 0

        while queue:
            u = queue.popleft()
            for v in edges[u]:
                if dist[v] == -1 and v <= group_size:
                    dist[v] = dist[u] + 1
                    queue.append(v)

        return dist

    dist_from_1 = bfs(1, N_1)
    dist_from_N = bfs(N_1 + N_2, N_1 + N_2)

    max_d = 1

    for u in range(1, N_1 + 1):
        for v in range(N_1 + 1, N_1 + N_2 + 1):
            if dist_from_1[u] != -1 and dist_from_N[v] != -1:
                max_d = max(max_d, dist_from_1[u] + dist_from_N[v] + 1)

    print(max_d)

if __name__ == "__main__":
    main()