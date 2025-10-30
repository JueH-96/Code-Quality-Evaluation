import sys
from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Increase the recursion limit for deep trees, which can occur for n up to 10^5.
        # A small buffer is added for safety.
        if n + 10 > sys.getrecursionlimit():
            sys.setrecursionlimit(n + 10)

        # 1. Build the adjacency list. Children are added in increasing order of index.
        adj = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p != -1:
                adj[p].append(i)

        # 2. Hashing parameters. Using two hashes minimizes collisions.
        p1, mod1 = 31, 10**9 + 7
        p2, mod2 = 37, 10**9 + 9

        # Precompute powers of p1 and p2.
        p1_powers = [1] * (n + 1)
        p2_powers = [1] * (n + 1)
        for i in range(1, n + 1):
            p1_powers[i] = (p1_powers[i - 1] * p1) % mod1
            p2_powers[i] = (p2_powers[i - 1] * p2) % mod2

        # 3. Storage for DFS results for each node u:
        #    - sizes[u]: Size of the subtree rooted at u.
        #    - hf1/hr1, hf2/hr2: Forward and reverse hashes.
        answer = [False] * n
        sizes = [0] * n
        hf1, hr1 = [0] * n, [0] * n
        hf2, hr2 = [0] * n, [0] * n

        # 4. Post-order DFS to compute hashes and check for palindromes.
        def dfs(u: int):
            # Recurse on children first.
            for v in adj[u]:
                dfs(v)
            
            # --- Combine results from processed children ---
            # Hashes for the concatenated string from children's DFS strings.
            h_C_fwd1, h_C_rev1 = 0, 0
            h_C_fwd2, h_C_rev2 = 0, 0
            size_C = 0

            # Forward pass to build the forward hash of combined children strings.
            for v in adj[u]:
                size_v = sizes[v]
                h_C_fwd1 = (h_C_fwd1 * p1_powers[size_v] + hf1[v]) % mod1
                h_C_fwd2 = (h_C_fwd2 * p2_powers[size_v] + hf2[v]) % mod2
                size_C += size_v
            
            # Reverse pass for the reverse hash of combined children strings.
            for v in reversed(adj[u]):
                size_v = sizes[v]
                h_C_rev1 = (h_C_rev1 * p1_powers[size_v] + hr1[v]) % mod1
                h_C_rev2 = (h_C_rev2 * p2_powers[size_v] + hr2[v]) % mod2

            # --- Compute properties for the current node u ---
            sizes[u] = size_C + 1
            val = ord(s[u]) - ord('a') + 1

            # Forward hash for dfsStr(u) = (children's strings) + s[u]
            hf1[u] = (h_C_fwd1 * p1 + val) % mod1
            hf2[u] = (h_C_fwd2 * p2 + val) % mod2

            # Reverse hash for dfsStr(u) = s[u] + (reversed children's strings)
            hr1[u] = (val * p1_powers[size_C] + h_C_rev1) % mod1
            hr2[u] = (val * p2_powers[size_C] + h_C_rev2) % mod2
            
            # Check for palindrome by comparing forward and reverse hashes.
            if hf1[u] == hr1[u] and hf2[u] == hr2[u]:
                answer[u] = True

        # 5. Start DFS from the root (node 0).
        dfs(0)
        
        return answer