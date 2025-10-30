import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        a = int(input[idx])
        b = int(input[idx+1])
        adj[a].append(b)
        adj[b].append(a)
        idx += 2
    C = [0] + [int(x) for x in input[idx:idx+N]]
    idx += N
    S = sum(C)
    
    sum_subtree = [0] * (N+1)
    f = [0] * (N+1)
    
    # First DFS to compute sum_subtree and f(1)
    stack = [(1, -1, 0, False)]
    f_total = 0
    while stack:
        node, parent, depth, processed = stack.pop()
        if not processed:
            stack.append((node, parent, depth, True))
            for neighbor in adj[node]:
                if neighbor != parent:
                    stack.append((neighbor, node, depth + 1, False))
        else:
            sum_subtree[node] = C[node]
            for neighbor in adj[node]:
                if neighbor != parent:
                    sum_subtree[node] += sum_subtree[neighbor]
            f_total += C[node] * depth
    f[1] = f_total
    
    # Second DFS to compute f(x) for all x
    stack = [(1, -1)]
    while stack:
        node, parent = stack.pop()
        for neighbor in adj[node]:
            if neighbor != parent:
                f[neighbor] = f[node] + S - 2 * sum_subtree[neighbor]
                stack.append((neighbor, node))
    
    # Find the minimum f(x)
    min_f = float('inf')
    for x in range(1, N+1):
        if f[x] < min_f:
            min_f = f[x]
    print(min_f)

if __name__ == '__main__':
    main()