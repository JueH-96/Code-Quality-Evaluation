import sys

def main():
    N, Q = map(int, sys.stdin.readline().split())
    graph = [[] for _ in range(N + 1)]

    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().split()))
        if query[0] == 1:
            u, v = query[1], query[2]
            graph[u].append(v)
            graph[v].append(u)
        else:
            v, k = query[1], query[2]
            connected_vertices = sorted(set(graph[v]), reverse=True)
            if k > len(connected_vertices):
                print(-1)
            else:
                print(connected_vertices[k - 1])

if __name__ == "__main__":
    main()