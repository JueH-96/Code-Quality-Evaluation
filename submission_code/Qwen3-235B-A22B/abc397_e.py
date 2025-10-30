import sys
sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N, K = int(input[idx]), int(input[idx+1])
    idx += 2
    NK = N * K

    if K == 1:
        print("Yes")
        return

    from collections import defaultdict

    adj = defaultdict(list)
    for _ in range(NK - 1):
        u = int(input[idx])
        v = int(input[idx + 1])
        adj[u].append(v)
        adj[v].append(u)
        idx += 2

    root = 1
    parent = [0] * (NK + 1)
    visited = [False] * (NK + 1)
    stack = [(root, False)]
    residue = [0] * (NK + 1)
    valid = True

    while stack and valid:
        node, processed = stack.pop()
        if not processed:
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            # Push unvisited children
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    parent[neighbor] = node
                    stack.append((neighbor, False))
        else:
            s = 1
            count_non_zero = 0
            for neighbor in adj[node]:
                if neighbor != parent[node]:
                    if parent[node] == neighbor:
                        continue
                    child_res = residue[neighbor]
                    s += child_res
                    if child_res != 0:
                        count_non_zero += 1
            if count_non_zero > 2:
                valid = False
                return
            mod = s % K
            residue[node] = mod

    if not valid or residue[root] % K != 0:
        print("No")
    else:
        print("Yes")

if __name__ == '__main__':
    main()