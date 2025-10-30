from typing import List

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        MOD = 10**9 + 7
        B = 911382629  # Choose a large base
        n = len(parent)
        
        # Precompute the power table
        pow_table = [1] * (n + 2)
        for i in range(1, n + 2):
            pow_table[i] = (pow_table[i - 1] * B) % MOD
        
        # Build the tree with sorted children
        tree = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            tree[p].append(i)
        for i in range(n):
            tree[i].sort()
        
        # Compute len_, forward_hash, reverse_hash using post-order traversal
        len_ = [0] * n
        forward_hash = [0] * n
        reverse_hash = [0] * n
        
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reverse order to process them left to right
                for child in reversed(tree[node]):
                    stack.append((child, False))
            else:
                # Compute len_ and hashes
                total_len = 1
                curr_forward = 0
                curr_rev = 0
                
                # Process children in order for forward hash
                for child in tree[node]:
                    curr_forward = (curr_forward * pow_table[len_[child]] + forward_hash[child]) % MOD
                    total_len += len_[child]
                
                # Add current character to forward hash
                char_val = ord(s[node]) - ord('a') + 1
                curr_forward = (curr_forward * B + char_val) % MOD
                
                # Process children in reverse order for reverse hash
                curr_rev = char_val
                for child in reversed(tree[node]):
                    curr_rev = (curr_rev * pow_table[len_[child]] + reverse_hash[child]) % MOD
                
                # Assign computed values
                forward_hash[node] = curr_forward
                reverse_hash[node] = curr_rev
                len_[node] = total_len
        
        # Generate the answer by comparing hashes
        answer = [forward_hash[i] == reverse_hash[i] for i in range(n)]
        return answer