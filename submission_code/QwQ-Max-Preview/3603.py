class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        mod = 10**18 + 3
        base = 911382629
        
        # Precompute base powers up to n
        max_pow = n
        base_pows = [1] * (max_pow + 1)
        for i in range(1, max_pow + 1):
            base_pows[i] = (base_pows[i-1] * base) % mod
        
        # Build children list for each node
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        for i in range(n):
            children[i].sort()
        
        # Post-order traversal
        post_order = []
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reversed order to process them in sorted order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                post_order.append(node)
        
        # Arrays to store hash values and lengths
        forward = [0] * n
        backward = [0] * n
        length = [0] * n
        answer = [False] * n
        
        # Process each node in post-order
        for node in post_order:
            sum_len = 0
            for child in children[node]:
                sum_len += length[child]
            
            # Compute combined forward hash
            combined_forward = 0
            for child in children[node]:
                child_len = length[child]
                combined_forward = (combined_forward * base_pows[child_len] + forward[child]) % mod
            
            # Compute combined backward hash
            combined_backward = 0
            for child in reversed(children[node]):
                child_len = length[child]
                combined_backward = (combined_backward * base_pows[child_len] + backward[child]) % mod
            
            # Current character hash
            char_hash = ord(s[node]) - ord('a') + 1
            
            # Update forward and backward hashes for current node
            forward[node] = (combined_forward * base + char_hash) % mod
            backward[node] = (char_hash * base_pows[sum_len] + combined_backward) % mod
            length[node] = sum_len + 1
            
            # Check if palindrome
            answer[node] = (forward[node] == backward[node])
        
        return answer