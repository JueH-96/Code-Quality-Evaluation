import sys
from sys import stdin
MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    T = int(data[ptr])
    ptr += 1
    
    for _ in range(T):
        N = int(data[ptr])
        ptr += 1
        
        p = list(map(int, data[ptr:ptr+N]))
        ptr += N
        
        a = list(map(int, data[ptr:ptr+N]))
        ptr += N
        
        # Build the tree
        children = [[] for _ in range(N+1)]  # 1-based
        for i in range(1, N+1):
            parent = p[i-1]
            children[parent].append(i)
        
        # Compute in_time and out_time using iterative DFS
        in_time = [0] * (N+1)
        out_time = [0] * (N+1)
        time = 0
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if visited:
                out_time[node] = time
                time += 1
            else:
                in_time[node] = time
                time += 1
                stack.append((node, True))
                # Push children in reverse order to process them in order
                for child in reversed(children[node]):
                    stack.append((child, False))
        
        # Compute S_u for each node
        S = [0] * (N+1)
        for u in range(1, N+1):
            sum_a = 0
            # Traverse the children in the order they were added
            # Using the in_time and children list
            # For each child, add a[child-1] to sum_a
            sum_a = 0
            stack = [u]
            while stack:
                node = stack.pop()
                sum_a += a[node-1]
                for child in children[node]:
                    stack.append(child)
            S[u] = sum_a
        
        # Compute inv_S
        inv_S = [0] * (N+1)
        for u in range(1, N+1):
            if S[u] == 0:
                inv_S[u] = 0
            else:
                inv_S[u] = pow(S[u], MOD-2, MOD)
        
        # Compute product and depth
        product = [0] * (N+1)
        depth = [0] * (N+1)
        product[0] = 1
        for u in range(1, N+1):
            product[u] = product[p[u-1]] * a[u-1] % MOD
            product[u] = product[u] * inv_S[p[u-1]] % MOD
        
        # Compute expected value
        expected = 0
        for v in range(1, N+1):
            expected = (expected + depth[v] * product[v]) % MOD
        
        print(expected % MOD)

if __name__ == '__main__':
    main()