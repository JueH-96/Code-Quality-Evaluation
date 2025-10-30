import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    adj = [set() for _ in range(N + 1)]
    degree = [0] * (N + 1)
    isolated = N

    for _ in range(Q):
        query = input[ptr]
        ptr += 1
        if query == '1':
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            if degree[u] == 0:
                isolated -= 1
            if degree[v] == 0:
                isolated -= 1
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1
        else:
            v = int(input[ptr])
            ptr += 1
            if degree[v] > 0:
                isolated += 1
                for u in list(adj[v]):
                    adj[v].remove(u)
                    adj[u].remove(v)
                    degree[u] -= 1
                    if degree[u] == 0:
                        isolated += 1
                degree[v] = 0
        print(isolated)

if __name__ == "__main__":
    main()