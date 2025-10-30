import sys
from collections import defaultdict

MOD = 998244353

def inv(x, mod):
    return pow(x, mod-2, mod)

def main():
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
    
    # Initialize parent and size for each player
    parent = [i for i in range(N+1)]
    size = [1] * (N+1)
    
    # Initialize expected wins for each player
    E = [0] * (N+1)
    
    for p, q in matches:
        # Find the roots of p and q
        root_p = p
        while parent[root_p] != root_p:
            root_p = parent[root_p]
        root_q = q
        while parent[root_q] != root_q:
            root_q = parent[root_q]
        
        if root_p == root_q:
            continue
        
        # Calculate the probabilities
        a = size[root_p]
        b = size[root_q]
        total = a + b
        prob_p = a * inv(total, MOD) % MOD
        prob_q = b * inv(total, MOD) % MOD
        
        # Update expected wins
        E[root_p] = (E[root_p] + prob_p) % MOD
        E[root_q] = (E[root_q] + prob_q) % MOD
        
        # Merge the two teams
        if size[root_p] < size[root_q]:
            parent[root_p] = root_q
            size[root_q] += size[root_p]
            # Propagate the expected wins to the new root
            E[root_q] = (E[root_q] + E[root_p]) % MOD
        else:
            parent[root_q] = root_p
            size[root_p] += size[root_q]
            # Propagate the expected wins to the new root
            E[root_p] = (E[root_p] + E[root_q]) % MOD
    
    # Now, for each player, find its root and get the expected wins
    result = []
    for i in range(1, N+1):
        root = i
        while parent[root] != root:
            root = parent[root]
        result.append(E[root])
    
    print(' '.join(map(str, result[1:])))

if __name__ == "__main__":
    main()