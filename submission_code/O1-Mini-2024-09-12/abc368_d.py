def solve():
    import sys
    from sys import stdin
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    K = int(input[ptr]); ptr +=1
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        A = int(input[ptr]); ptr +=1
        B = int(input[ptr]); ptr +=1
        adj[A].append(B)
        adj[B].append(A)
    V = set()
    for _ in range(K):
        V.add(int(input[ptr])); ptr +=1
    necessary = [False]*(N+1)
    stack = [(1, 0, False)]
    while stack:
        node, parent, visited = stack.pop()
        if visited:
            if node in V:
                necessary[node] = True
            else:
                for child in adj[node]:
                    if child != parent and necessary[child]:
                        necessary[node] = True
                        break
        else:
            stack.append((node, parent, True))
            for child in adj[node]:
                if child != parent:
                    stack.append((child, node, False))
    count = sum(necessary)
    print(count)