from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(parent)
        # build children lists
        children = [[] for _ in range(n)]
        for i, p in enumerate(parent):
            if p >= 0:
                children[p].append(i)
        
        # idx[u] = post‐order index of u in the global traversal
        # L[u]   = the minimum post‐order index in u's subtree
        idx = [0] * n
        L   = [0] * n
        cnt = 0
        
        def dfs(u):
            nonlocal cnt
            min_sub = float('inf')
            for v in children[u]:
                dfs(v)
                min_sub = min(min_sub, L[v])
            cur = cnt
            idx[u] = cur
            cnt += 1
            if min_sub == float('inf'):
                # leaf
                min_sub = cur
            L[u] = min_sub
        
        # do one global post‐order from root=0
        dfs(0)
        
        # build the character array in post‐order
        arr = [0] * n
        for u in range(n):
            arr[idx[u]] = ord(s[u]) - ord('a') + 1
        
        # reverse it for reverse‐hash
        arr_rev = arr[::-1]
        
        # hash parameters
        mod1, mod2 = 10**9+7, 10**9+9
        base = 9113823
        
        # precompute powers
        p1 = [1] * (n+1)
        p2 = [1] * (n+1)
        for i in range(n):
            p1[i+1] = (p1[i] * base) % mod1
            p2[i+1] = (p2[i] * base) % mod2
        
        # forward prefix hashes on arr
        hf1 = [0] * (n+1)
        hf2 = [0] * (n+1)
        for i in range(n):
            hf1[i+1] = (hf1[i] * base + arr[i]) % mod1
            hf2[i+1] = (hf2[i] * base + arr[i]) % mod2
        
        # forward prefix hashes on arr_rev
        hr1 = [0] * (n+1)
        hr2 = [0] * (n+1)
        for i in range(n):
            hr1[i+1] = (hr1[i] * base + arr_rev[i]) % mod1
            hr2[i+1] = (hr2[i] * base + arr_rev[i]) % mod2
        
        # answer queries
        ans = [False] * n
        for u in range(n):
            l = L[u]
            r = idx[u]
            length = r - l + 1
            
            # hash of arr[l..r]
            x1 = (hf1[r+1] - hf1[l] * p1[length]) % mod1
            x2 = (hf2[r+1] - hf2[l] * p2[length]) % mod2
            
            # corresponding reversed segment in arr_rev is
            # indices [n-1-r .. n-1-l]
            rl = n - 1 - r
            rr = n - 1 - l
            y1 = (hr1[rr+1] - hr1[rl] * p1[length]) % mod1
            y2 = (hr2[rr+1] - hr2[rl] * p2[length]) % mod2
            
            if x1 == y1 and x2 == y2:
                ans[u] = True
        
        return ans