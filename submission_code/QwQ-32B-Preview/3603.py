class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        from collections import defaultdict

        MOD1 = 10**9 + 7
        MOD2 = 10**9 + 9
        P = 31

        # Build adjacency list
        adj = defaultdict(list)
        for i in range(1, len(parent)):
            adj[parent[i]].append(i)
        for node in adj:
            adj[node].sort()

        # Initialize answer list
        answer = [False] * len(parent)

        # DFS function to compute hashes
        def dfs(node):
            # Base case: leaf node
            if not adj[node]:
                hash_forward = ord(s[node]) - ord('a') + 1
                hash_backward = hash_forward
                return (hash_forward, hash_backward, 1)
            
            # Initialize hashes and total length
            hash_forward = 0
            hash_backward = 0
            total_len = 0

            # Process each child
            for child in adj[node]:
                child_forward, child_backward, child_len = dfs(child)
                # Update forward hash
                hash_forward = (hash_forward * P + child_forward) % MOD1
                # Update backward hash
                mul = pow(P, child_len, MOD2)
                hash_backward = (child_backward * mul + hash_backward) % MOD2
                # Update total length
                total_len += child_len
            
            # Append current node's character
            char_code = ord(s[node]) - ord('a') + 1
            hash_forward = (hash_forward * P + char_code) % MOD1
            hash_backward = (char_code * P**total_len + hash_backward) % MOD2
            total_len += 1

            # Check if hashes match
            if hash_forward == hash_backward:
                answer[node] = True
            else:
                answer[node] = False
            
            return (hash_forward, hash_backward, total_len)
        
        # Start DFS from each node
        for i in range(len(parent)):
            dfs(i)
        
        return answer