MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    matches = []
    index = 1
    for _ in range(N-1):
        p = int(data[index])
        q = int(data[index+1])
        matches.append((p, q))
        index += 2
    
    # Initialize parent and size for Union-Find
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    
    # Initialize expected wins for each player
    E = [0] * (N+1)
    
    for p, q in matches:
        # Find the roots of p and q
        root_p = find(parent, p)
        root_q = find(parent, q)
        
        # Calculate the probabilities
        a = size[root_p]
        b = size[root_q]
        prob_p = a * pow(a + b, MOD-2, MOD) % MOD
        prob_q = b * pow(a + b, MOD-2, MOD) % MOD
        
        # Update expected wins
        E[root_p] = (E[root_p] + prob_p) % MOD
        E[root_q] = (E[root_q] + prob_q) % MOD
        
        # Union the two sets
        if root_p != root_q:
            if size[root_p] < size[root_q]:
                parent[root_p] = root_q
                size[root_q] += size[root_p]
                E[root_q] = (E[root_q] + E[root_p]) % MOD
            else:
                parent[root_q] = root_p
                size[root_p] += size[root_q]
                E[root_p] = (E[root_p] + E[root_q]) % MOD
    
    # Now, for each player, find their root and get the expected wins
    result = []
    for i in range(1, N+1):
        root = find(parent, i)
        result.append(str(E[root]))
    
    print(' '.join(result))

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

if __name__ == "__main__":
    main()