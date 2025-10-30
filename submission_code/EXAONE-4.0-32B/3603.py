from collections import deque
from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        mod1 = 10**9 + 7
        mod2 = 10**9 + 9
        base1 = 131
        base2 = 137
        
        max_len = n
        pow1 = [1] * (max_len + 1)
        pow2 = [1] * (max_len + 1)
        for i in range(1, max_len + 1):
            pow1[i] = (pow1[i-1] * base1) % mod1
            pow2[i] = (pow2[i-1] * base2) % mod2
        
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        for i in range(n):
            children[i].sort()
        
        deg = [len(children[i]) for i in range(n)]
        
        L = [0] * n
        H_forward1 = [0] * n
        H_forward2 = [0] * n
        H_backward1 = [0] * n
        H_backward2 = [0] * n
        ans = [False] * n
        
        q = deque()
        for i in range(n):
            if deg[i] == 0:
                q.append(i)
                
        while q:
            u = q.popleft()
            total_length_children = 0
            f1, f2 = 0, 0
            for v in children[u]:
                total_length_children += L[v]
                f1 = (f1 * pow1[L[v]] + H_forward1[v]) % mod1
                f2 = (f2 * pow2[L[v]] + H_forward2[v]) % mod2
            
            L[u] = total_length_children + 1
            f1 = (f1 * base1 + ord(s[u])) % mod1
            f2 = (f2 * base2 + ord(s[u])) % mod2
            H_forward1[u] = f1
            H_forward2[u] = f2
            
            b1, b2 = ord(s[u]), ord(s[u])
            for v in reversed(children[u]):
                b1 = (b1 * pow1[L[v]] + H_backward1[v]) % mod1
                b2 = (b2 * pow2[L[v]] + H_backward2[v]) % mod2
            H_backward1[u] = b1
            H_backward2[u] = b2
            
            if f1 == b1 and f2 == b2:
                ans[u] = True
            else:
                ans[u] = False
                
            p = parent[u]
            if p != -1:
                deg[p] -= 1
                if deg[p] == 0:
                    q.append(p)
                    
        return ans