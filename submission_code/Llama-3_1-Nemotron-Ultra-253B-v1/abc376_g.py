import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr +=1
    for _ in range(T):
        N = int(input[ptr])
        ptr +=1
        p = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        a = list(map(int, input[ptr:ptr+N]))
        ptr +=N
        
        # Build the tree
        children = [[] for _ in range(N+1)]
        for i in range(N):
            u = i+1
            parent = p[i]
            children[parent].append(u)
        
        # Compute S and size
        S = [0]*(N+1)
        size = [1]*(N+1)
        stack = []
        visited = [False]*(N+1)
        for u in range(N+1):
            if not visited[u]:
                stack.append( (u, False) )
                while stack:
                    node, done = stack.pop()
                    if done:
                        for v in children[node]:
                            S[node] += S[v]
                            size[node] += size[v]
                        if node !=0:
                            S[node] += a[node-1]
                    else:
                        stack.append( (node, True) )
                        for v in reversed(children[node]):
                            stack.append( (v, False) )
        
        # Sort children by S in descending order
        sorted_children = [[] for _ in range(N+1)]
        for u in range(N+1):
            sorted_children[u] = sorted(children[u], key=lambda x: (-S[x], x))
        
        # Assign step numbers using iterative pre-order traversal
        step = [0]*(N+1)
        current_step = 1
        stack = []
        # Process root's children in sorted order
        for v in reversed(sorted_children[0]):
            stack.append( (v, False) )
        
        while stack:
            node, visited_flag = stack.pop()
            if not visited_flag:
                step[node] = current_step
                current_step +=1
                stack.append( (node, True) )
                # Push children in reversed sorted order
                for child in reversed(sorted_children[node]):
                    stack.append( (child, False) )
        
        # Compute sum_contribution
        sum_contribution = 0
        for i in range(N):
            u = i+1
            sum_contribution = (sum_contribution + a[i] * step[u]) % MOD
        
        S_total = sum(a) % MOD
        inv_S_total = pow(S_total, MOD-2, MOD)
        ans = (sum_contribution * inv_S_total) % MOD
        print(ans)

if __name__ == '__main__':
    main()