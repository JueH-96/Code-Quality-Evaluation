class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        if n == 0:
            return []
        
        # Build the tree structure with children sorted
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        for node in range(n):
            children[node].sort()
        
        # Precompute powers of the base modulo mod up to the maximum needed length
        mod = 10**18 + 3
        base = 911382629
        max_pow = n  # Maximum possible length is n for the root node
        pow_base = [1] * (max_pow + 1)
        for i in range(1, max_pow + 1):
            pow_base[i] = (pow_base[i-1] * base) % mod
        
        # Initialize arrays to store hash_forward and hash_reverse for each node
        hash_forward = [(0, 0)] * n  # (hash_value, length)
        hash_reverse = [(0, 0)] * n  # (hash_value, length)
        
        # Iterative post-order traversal using a stack
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reverse order to process them in sorted order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                # Compute hash_forward for the current node
                hf_h = 0
                hf_l = 0
                for child in children[node]:
                    ch_h, ch_l = hash_forward[child]
                    hf_h = (hf_h * pow_base[ch_l] + ch_h) % mod
                    hf_l += ch_l
                # Append the current node's character
                hf_h = (hf_h * base + ord(s[node])) % mod
                hf_l += 1
                hash_forward[node] = (hf_h, hf_l)
                
                # Compute hash_reverse for the current node
                hr_h = ord(s[node])
                hr_l = 1
                # Process children in reverse order (since children are sorted)
                for child in reversed(children[node]):
                    ch_h_rev, ch_l_rev = hash_reverse[child]
                    hr_h = (hr_h * pow_base[ch_l_rev] + ch_h_rev) % mod
                    hr_l += ch_l_rev
                hash_reverse[node] = (hr_h, hr_l)
        
        # Determine the answer for each node
        answer = [False] * n
        for i in range(n):
            if hash_forward[i][0] == hash_reverse[i][0]:
                answer[i] = True
            else:
                answer[i] = False
        
        return answer