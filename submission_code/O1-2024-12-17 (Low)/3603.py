class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(parent)
        if n == 1:
            return [True]  # Single-character string is always a palindrome
        
        # Build adjacency list (children array) for each node
        # Note: parent[i] = x means x is the parent of i
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        
        # We need to visit children in ascending order
        for i in range(n):
            children[i].sort()
        
        # We'll perform a single postorder traversal from the root (0).
        # However, we need a valid tree with root=0 (parent[0] = -1).
        # We record the order in which nodes are placed in postorder,
        # along with subtree ranges (start[i], end[i]) in that postorder array.
        
        postorder = [0] * n   # will store the node-ids in postorder order
        start = [0] * n       # start[i] index of subtree of i in postorder array
        end = [0] * n         # end[i]   index of subtree of i in postorder array
        idx = 0               # global index over postorder array
        
        def dfs(x: int) -> None:
            nonlocal idx
            local_start = idx
            # Recurse on children first
            for c in children[x]:
                dfs(c)
            # Put x into postorder array
            postorder[idx] = x
            idx += 1
            # The entire subtree of x occupies [local_start..idx-1]
            start[x] = local_start
            end[x] = idx - 1
        
        # Call dfs from the root (node 0)
        dfs(0)
        
        # Now postorder array gives us the nodes in postorder.
        # We need a quick way to check if the substring of characters
        # for each subtree (start[x]..end[x]) is a palindrome.
        # We'll build a rolling-hash (forward and backward) on the characters
        # in the postorder array. For each position i in [0..n-1],
        # let char = s[ postorder[i] ].
        
        # Build the array of characters in postorder
        chars = [s[postorder[i]] for i in range(n)]
        
        # Polynomial rolling hash parameters
        base = 257
        mod = 10**9 + 7
        
        # Precompute powers of base
        power = [1] * (n+1)
        for i in range(n):
            power[i+1] = (power[i] * base) % mod
        
        # Forward hash: forwardHash[i+1] = hash of chars[0..i] (inclusive)
        forwardHash = [0] * (n+1)
        for i in range(n):
            forwardHash[i+1] = (forwardHash[i] * base + (ord(chars[i]) - ord('a') + 1)) % mod
        
        # Backward hash: backwardHash[i] = hash of chars[i..n-1] (inclusive)
        backwardHash = [0] * (n+1)
        backwardHash[n] = 0
        for i in range(n-1, -1, -1):
            backwardHash[i] = (backwardHash[i+1] * base + (ord(chars[i]) - ord('a') + 1)) % mod
        
        def get_forward_hash(l: int, r: int) -> int:
            # substring chars[l..r], inclusive
            length = r - l + 1
            val = forwardHash[r+1] - (forwardHash[l] * power[length] % mod)
            return val % mod
        
        def get_backward_hash(l: int, r: int) -> int:
            # substring chars[l..r], inclusive
            length = r - l + 1
            val = backwardHash[l] - (backwardHash[r+1] * power[length] % mod)
            return val % mod
        
        # Now for each node i, the postorder-subtree is [start[i]..end[i]].
        # Check if that substring is a palindrome via comparing forward/hash.
        
        answer = [False] * n
        for i in range(n):
            l = start[i]
            r = end[i]
            if get_forward_hash(l, r) == get_backward_hash(l, r):
                answer[i] = True
        
        return answer