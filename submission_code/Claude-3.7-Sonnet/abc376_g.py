# YOUR CODE HERE
def solve():
    T = int(input())
    for _ in range(T):
        N = int(input())
        parents = list(map(int, input().split()))
        a = list(map(int, input().split()))
        
        # Construct the tree
        children = [[] for _ in range(N+1)]
        for i in range(N):
            children[parents[i]].append(i+1)
        
        # Initially, only the root is searched
        searched = [False] * (N+1)
        searched[0] = True
        
        # Track the search position of each vertex
        pos = [0] * (N+1)
        
        # Greedily search vertices
        position = 1
        ready = children[0][:]  # Vertices whose parent is the root
        
        while ready and position <= N:
            # Choose the vertex with the highest probability
            chosen_idx = 0
            for i in range(1, len(ready)):
                if a[ready[i]-1] > a[ready[chosen_idx]-1]:
                    chosen_idx = i
            
            chosen = ready.pop(chosen_idx)
            pos[chosen] = position
            searched[chosen] = True
            position += 1
            
            # Add the unsearched children of the chosen vertex to the ready list
            for child in children[chosen]:
                if not searched[child]:
                    ready.append(child)
        
        # Compute the expected number of operations
        numerator = 0
        denominator = sum(a)
        
        for i in range(1, N+1):
            numerator += pos[i] * a[i-1]
        
        # Reduce the fraction
        from math import gcd
        gcd_val = gcd(numerator, denominator)
        numerator //= gcd_val
        denominator //= gcd_val
        
        # Compute modular inverse
        mod = 998244353
        inverse = pow(denominator, mod - 2, mod)
        result = (numerator * inverse) % mod
        
        print(result)

solve()