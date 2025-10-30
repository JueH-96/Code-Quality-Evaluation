class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(parent)
        
        # Build adjacency list (children) and sort children to respect increasing order
        children = [[] for _ in range(n)]
        for i in range(n):
            if parent[i] != -1:
                children[parent[i]].append(i)
        for ch in children:
            ch.sort()
        
        # Prepare arrays to store post-order traversal info
        post = []       # will hold nodes in post-order
        out = [0]*n     # out[node] = index of 'node' in 'post'
        size_subtree = [0]*n
        
        # Post-order DFS from root (0)
        def dfs(u: int) -> None:
            for v in children[u]:
                dfs(v)
                size_subtree[u] += size_subtree[v]
            post.append(u)
            out[u] = len(post) - 1
            size_subtree[u] += 1
        
        # Run DFS from the root
        dfs(0)
        
        # Build a string that follows the post-order of all nodes
        postS = ''.join(s[node] for node in post)
        
        # We will use a rolling-hash approach to check for palindromes in O(1) per query
        
        # Define parameters for polynomial rolling hash
        base = 257
        mod = 10**9 + 7
        
        # Precompute powers of base
        p = [1] * (n + 1)
        for i in range(n):
            p[i+1] = (p[i] * base) % mod
        
        # Forward hash array for postS
        F = [0] * (n + 1)
        for i, ch in enumerate(postS):
            F[i+1] = (F[i] * base + (ord(ch) - ord('a') + 1)) % mod
        
        # Build the reverse of postS and its forward hash array
        R = postS[::-1]
        G = [0] * (n + 1)
        for i, ch in enumerate(R):
            G[i+1] = (G[i] * base + (ord(ch) - ord('a') + 1)) % mod
        
        # Helper to get hash of a substring [l, r] (0-based) in a prefix-hash array H
        def get_hash(H, l, r):
            h = H[r+1] - (H[l] * p[r - l + 1] % mod)
            return h % mod
        
        # For each node i, check if the post-order subtree string is a palindrome
        answer = [False] * n
        for i in range(n):
            length = size_subtree[i]
            r = out[i]              # end of substring in postS
            l = r - length + 1      # start of substring in postS
            
            # Forward hash of postS[l..r]
            fwd_hash = get_hash(F, l, r)
            
            # Corresponding substring in R is from (n-1-r) to (n-1-l) inclusive
            rev_l = n - 1 - r
            rev_r = n - 1 - l
            bwd_hash = get_hash(G, rev_l, rev_r)
            
            answer[i] = (fwd_hash == bwd_hash)
        
        return answer