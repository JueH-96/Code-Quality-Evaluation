from typing import List
import sys
sys.setrecursionlimit(10**7)

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        
        # Build adjacency list of children (since parent[0] = -1, node 0 is root)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        # Sort children so DFS visits them in ascending order
        for i in range(n):
            children[i].sort()
        
        # Post-order DFS from the root (0) to get a global post-order array
        post_order = []
        def dfs_post(x):
            for c in children[x]:
                dfs_post(c)
            post_order.append(x)
        
        dfs_post(0)
        
        # tin[x] = index of node x in the post_order array
        tin = [0]*n
        for i, node in enumerate(post_order):
            tin[node] = i
        
        # We'll compute st[x] = left bound of the subtree of x in post_order
        # and en[x] = right bound (which is tin[x]) in post_order.
        st = [0]*n
        en = [0]*n
        
        def compute_st_en(x):
            st[x] = tin[x]
            for c in children[x]:
                compute_st_en(c)
                st[x] = min(st[x], st[c])
            en[x] = tin[x]
        
        compute_st_en(0)
        
        # Build the array T of length n (the post-order characters)
        # T[i] = s[ post_order[i] ], stored as integer 1..26 for hashing
        T = [ord(s[node]) - ord('a') + 1 for node in post_order]
        
        # Build the reversed array R of T for reverse-hashing
        R = T[::-1]
        
        # Prepare for polynomial rolling hash (double hashing to reduce collisions)
        mod1, mod2 = 10**9+7, 10**9+9
        base = 137
        
        # Prefix hashes for T (forward) and R (reverse)
        F1 = [0]*(n+1)
        F2 = [0]*(n+1)
        FR1 = [0]*(n+1)
        FR2 = [0]*(n+1)
        
        # Precompute powers of base
        P1 = [1]*(n+1)
        P2 = [1]*(n+1)
        
        for i in range(n):
            F1[i+1] = (F1[i]*base + T[i]) % mod1
            F2[i+1] = (F2[i]*base + T[i]) % mod2
            FR1[i+1] = (FR1[i]*base + R[i]) % mod1
            FR2[i+1] = (FR2[i]*base + R[i]) % mod2
            P1[i+1] = (P1[i]*base) % mod1
            P2[i+1] = (P2[i]*base) % mod2
        
        # Get forward hash for substring T[l..r]
        def get_hash_f(l, r):
            length = r - l + 1
            h1 = (F1[r+1] - (F1[l] * P1[length] % mod1) + mod1) % mod1
            h2 = (F2[r+1] - (F2[l] * P2[length] % mod2) + mod2) % mod2
            return (h1, h2)
        
        # Get forward hash for substring R[l..r]
        def get_hash_r(l, r):
            length = r - l + 1
            h1 = (FR1[r+1] - (FR1[l] * P1[length] % mod1) + mod1) % mod1
            h2 = (FR2[r+1] - (FR2[l] * P2[length] % mod2) + mod2) % mod2
            return (h1, h2)
        
        # Check if T[l..r] is a palindrome by comparing to its reverse substring in R
        def is_palindrome(l, r):
            # T[l..r] reversed corresponds to R[n-1-r .. n-1-l] in forward order
            L = n - 1 - r
            R_ = n - 1 - l
            return get_hash_f(l, r) == get_hash_r(L, R_)
        
        # For each node x, check if the post-order subtree T[st[x]..en[x]] is palindrome
        answer = [False]*n
        for x in range(n):
            l, r = st[x], en[x]
            # l..r in T is the DFS-postorder-subtree of x
            answer[x] = is_palindrome(l, r)
        
        return answer