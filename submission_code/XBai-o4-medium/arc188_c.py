import sys

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N, M = map(int, input().split())
    size = 2 * N
    graph = [[] for _ in range(size)]
    reverse_graph = [[] for _ in range(size)]
    
    for _ in range(M):
        A, B, C = map(int, input().split())
        a_true = 2 * (A - 1)
        a_false = a_true + 1
        b_true = 2 * (B - 1)
        b_false = b_true + 1
        
        if C == 1:
            # Add edges for (¬A ∨ B)
            graph[a_true].append(b_true)
            reverse_graph[b_true].append(a_true)
            graph[b_false].append(a_false)
            reverse_graph[a_false].append(b_false)
            # Add edges for (¬B ∨ A)
            graph[b_true].append(a_true)
            reverse_graph[a_true].append(b_true)
            graph[a_false].append(b_false)
            reverse_graph[b_false].append(a_false)
        else:
            # Add edges for (A ∨ B)
            graph[a_false].append(b_true)
            reverse_graph[b_true].append(a_false)
            graph[b_false].append(a_true)
            reverse_graph[a_true].append(b_false)
            # Add edges for (¬A ∨ ¬B)
            graph[a_true].append(b_false)
            reverse_graph[b_false].append(a_true)
            graph[b_true].append(a_false)
            reverse_graph[a_false].append(b_true)
    
    # First pass to get the order
    visited = [False] * size
    order = []
    for i in range(size):
        if not visited[i]:
            stack = [(i, False)]
            while stack:
                node, processed = stack.pop()
                if processed:
                    order.append(node)
                    continue
                if visited[node]:
                    continue
                visited[node] = True
                stack.append((node, True))
                # Add children
                for v in graph[node]:
                    if not visited[v]:
                        stack.append((v, False))
    
    # Second pass to get components
    visited2 = [False] * size
    component = [0] * size
    current = 0
    for u in reversed(order):
        if not visited2[u]:
            stack = [u]
            visited2[u] = True
            component[u] = current
            while stack:
                node = stack.pop()
                for v in reverse_graph[node]:
                    if not visited2[v]:
                        visited2[v] = True
                        component[v] = current
                        stack.append(v)
            current += 1
    
    # Check for contradictions and build the result
    res = []
    for i in range(1, N + 1):
        var_true = 2 * (i - 1)
        var_false = var_true + 1
        if component[var_true] == component[var_false]:
            print(-1)
            return
        if component[var_true] > component[var_false]:
            res.append('1')
        else:
            res.append('0')
    print(''.join(res))

if __name__ == "__main__":
    main()