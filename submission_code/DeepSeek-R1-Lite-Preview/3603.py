class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        import sys
        sys.setrecursionlimit(1 << 25)
        n = len(parent)
        base = 911382629
        modulus = 10**18 + 3

        # Build the tree
        children = [[] for _ in range(n)]
        for child in range(1, n):
            parent_node = parent[child]
            children[parent_node].append(child)
        
        for child_list in children:
            child_list.sort()
        
        # Precompute base powers
        max_len = n
        power = [1] * (max_len + 1)
        for i in range(1, max_len + 1):
            power[i] = (power[i - 1] * base) % modulus
        
        # Initialize arrays
        len_traversal = [0] * n
        hash_forward = [0] * n
        hash_backward = [0] * n
        
        # Post-order traversal to compute len_traversal, hash_forward, hash_backward
        def dfs(x):
            # Compute len_traversal[x]
            total_len = 0
            for y in children[x]:
                dfs(y)
                total_len += len_traversal[y]
            len_traversal[x] = total_len + 1
            
            # Compute hash_forward[x]
            hash_forward[x] = 0
            current = 0
            for y in children[x]:
                current = (current * power[len_traversal[y]] + hash_forward[y]) % modulus
            hash_forward[x] = (current * base + ord(s[x])) % modulus
            
            # Compute hash_backward[x]
            hash_backward[x] = ord(s[x]) * power[total_len] % modulus
            factor = 1
            for y in reversed(children[x]):
                hash_backward[x] = (hash_backward[x] + hash_backward[y] * factor) % modulus
                factor = factor * power[len_traversal[y]] % modulus
            
        dfs(0)
        
        # Check for palindrome
        answer = [False] * n
        for x in range(n):
            answer[x] = (hash_forward[x] == hash_backward[x])
        
        return answer