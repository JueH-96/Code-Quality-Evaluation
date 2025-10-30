from typing import List
import sys
sys.setrecursionlimit(10**6)

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(s)
        mod = 10**9 + 7
        base = 131
        
        # Precompute powers up to n (maximum string length can be n)
        pows = [1] * (n + 1)
        for i in range(1, n+1):
            pows[i] = (pows[i-1] * base) % mod
        
        # Build children list from parent array.
        children = [[] for _ in range(n)]
        for i in range(1, n):
            par = parent[i]
            children[par].append(i)
        
        # Since DFS requires children in increasing order, sort them.
        for i in range(n):
            children[i].sort()
        
        # answer list to record if f(x) is palindrome.
        ans = [False] * n
        
        # recursive function to return (forward_hash, reverse_hash, length) for subtree rooted at x.
        def dfs(x: int):
            # For an empty aggregation from children, we will combine later with current node's char.
            # Because f(x) = ( f(child1) + ... + f(child_k) ) + s[x]
            total_len = 0
            forward = 0
            # Process children in increasing order.
            for child in children[x]:
                child_hash, child_rev, child_len = dfs(child)
                # Append child's hash (like concatenating a string)
                forward = (forward * pows[child_len] + child_hash) % mod
                total_len += child_len
            
            # Now, append current node's character at the end.
            cur_val = ord(s[x])
            forward = (forward * base + cur_val) % mod
            total_len += 1
            
            # Compute reverse hash.
            # Reverse(f(x)) = s[x] + reverse(f(child_k)) + ... + reverse(f(child₁)).
            rev = cur_val  # starting from s[x].
            # Process children in reversed order.
            for child in reversed(children[x]):
                child_hash, child_rev, child_len = dfs_results[child]
                rev = (rev * pows[child_len] + child_rev) % mod
            
            # Check if this subtree DFS string is palindrome.
            if forward == rev:
                ans[x] = True
            else:
                ans[x] = False
            
            # Save the computed triple for use in parent's reverse computation.
            dfs_results[x] = (forward, rev, total_len)
            return dfs_results[x]
        
        # We need to avoid recomputing dfs(child) twice.
        # We use an array to store results after computing a node.
        dfs_results = [None] * n
        
        # We run dfs on every node in one pass.
        # Since tree is rooted at 0, but we need answer for every node (subtree DFS string)
        # we perform a DFS starting from root 0.
        # However, note that the tree is not necessarily binary and might be "small"
        # so the typical one DFS from root 0 would compute dfs for every node exactly once.
        dfs(0)
        
        return ans