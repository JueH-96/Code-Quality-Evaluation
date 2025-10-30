from typing import List
import sys, random

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        sys.setrecursionlimit(300000)           # just in case
        
        n = len(parent)
        
        # ------------------------------------------------------------
        # 1. build the children lists (already sorted thanks to index order)
        # ------------------------------------------------------------
        children = [[] for _ in range(n)]
        for v in range(1, n):
            children[parent[v]].append(v)       # children are added in
                                                 # ascending order automatically
        
        # ------------------------------------------------------------
        # 2. produce the post-order of the whole tree
        #    together with  pos[v] – position of v in that order
        # ------------------------------------------------------------
        postorder = []
        stack = [(0, False)]                    # (node, visited_children?)
        while stack:
            node, done = stack.pop()
            if done:
                postorder.append(node)
            else:
                stack.append((node, True))      # come back to the node later
                # push children in *reverse* ascending order
                for c in reversed(children[node]):
                    stack.append((c, False))
        
        pos = [0]*n                             # position of every node
        for idx, node in enumerate(postorder):
            pos[node] = idx
        
        # ------------------------------------------------------------
        # 3. subtree sizes  (using post-order: children already processed)
        # ------------------------------------------------------------
        size = [1]*n
        for node in postorder:                  # children → parent
            p = parent[node]
            if p != -1:
                size[p] += size[node]
        
        # ------------------------------------------------------------
        # 4. build the character sequence of the post-order walk
        # ------------------------------------------------------------
        char_val = [ord(s[v]) - 96 for v in postorder]   # ‘a’ → 1, … ‘z’ → 26
        
        # ------------------------------------------------------------
        # 5. double rolling hashes for the sequence and its reverse
        # ------------------------------------------------------------
        m1, m2  = 1_000_000_007, 1_000_000_009
        b1, b2  = 911382323,     972663749     # large random bases < mod
        
        pow1 = [1]*(n+1)
        pow2 = [1]*(n+1)
        for i in range(1, n+1):
            pow1[i] = (pow1[i-1]*b1) % m1
            pow2[i] = (pow2[i-1]*b2) % m2
        
        pref1 = [0]*(n+1)                      # forward hash
        pref2 = [0]*(n+1)
        for i, val in enumerate(char_val, 1):
            pref1[i] = (pref1[i-1]*b1 + val) % m1
            pref2[i] = (pref2[i-1]*b2 + val) % m2
        
        rev_val = char_val[::-1]               # reversed sequence
        rpref1 = [0]*(n+1)
        rpref2 = [0]*(n+1)
        for i, val in enumerate(rev_val, 1):
            rpref1[i] = (rpref1[i-1]*b1 + val) % m1
            rpref2[i] = (rpref2[i-1]*b2 + val) % m2
        
        def substring_hash(pref, l, r, pow_, mod):
            """hash of inclusive [l, r] in O(1)"""
            res = pref[r+1] - (pref[l] * pow_[r-l+1] % mod)
            if res < 0:
                res += mod
            return res
        
        # ------------------------------------------------------------
        # 6. answer for every node
        # ------------------------------------------------------------
        ans = [False]*n
        for v in range(n):
            length = size[v]
            r = pos[v]
            l = r - length + 1                 # segment [l, r] of the big string
            
            # forward hashes
            h1  = substring_hash(pref1, l, r, pow1, m1)
            h1r = substring_hash(pref2, l, r, pow2, m2)
            
            # corresponding segment in the reversed string
            rl = n - 1 - r
            rr = n - 1 - l
            rh1  = substring_hash(rpref1, rl, rr, pow1, m1)
            rh1r = substring_hash(rpref2, rl, rr, pow2, m2)
            
            ans[v] = (h1 == rh1) and (h1r == rh1r)
        
        return ans