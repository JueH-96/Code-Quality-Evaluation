class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        if n == 0:
            return []
        
        # Build the tree structure
        tree = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            tree[p].append(i)
        
        # Sort each node's children in increasing order
        for i in range(n):
            tree[i].sort()
        
        # Choose base and modulus for polynomial rolling hash
        P = 911382629
        MOD = 10**18 + 3
        
        # Precompute powers of P up to the maximum possible length
        max_len = n
        pow_P = [1] * (max_len + 2)
        for i in range(1, max_len + 2):
            pow_P[i] = (pow_P[i-1] * P) % MOD
        
        # Perform iterative post-order traversal
        post_order = []
        stack = [(0, False)]
        while stack:
            node, processed = stack.pop()
            if not processed:
                stack.append((node, True))
                # Push children in reversed order to process them in correct order
                for child in reversed(tree[node]):
                    stack.append((child, False))
            else:
                post_order.append(node)
        
        # Initialize arrays to store hash_forward, hash_backward, and length of the string
        hash_forward = [0] * n
        hash_backward = [0] * n
        len_str = [0] * n
        
        for x in post_order:
            children = tree[x]
            
            # Compute hash_forward
            hf = 0
            for child in children:
                hf = (hf * pow_P[len_str[child]] + hash_forward[child]) % MOD
            hf = (hf * P + ord(s[x])) % MOD
            hash_forward[x] = hf
            
            # Compute length of the string
            ls = 0
            for child in children:
                ls += len_str[child]
            ls += 1
            len_str[x] = ls
            
            # Compute hash_backward
            hb = ord(s[x])
            for child in reversed(children):
                hb = (hb * pow_P[len_str[child]] + hash_backward[child]) % MOD
            hash_backward[x] = hb
        
        # Generate the answer list
        ans = [False] * n
        for i in range(n):
            if hash_forward[i] == hash_backward[i]:
                ans[i] = True
            else:
                ans[i] = False
        
        return ans