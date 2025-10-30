from typing import List
import sys
import sys
sys.setrecursionlimit(1 << 25)

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        root = -1
        for i in range(n):
            if parent[i] == -1:
                root = i
            else:
                children[parent[i]].append(i)
        
        # Sort children for each node in increasing order
        for child_list in children:
            child_list.sort()
        
        # Define two different hash functions to minimize collision
        mod1 = 10**9 + 7
        base1 = 911382629
        mod2 = 10**9 + 9
        base2 = 357142857
        
        # Precompute powers for both hash functions
        max_len = n + 1
        base_pows1 = [1] * (max_len + 1)
        base_pows2 = [1] * (max_len + 1)
        for i in range(1, max_len +1):
            base_pows1[i] = (base_pows1[i-1] * base1) % mod1
            base_pows2[i] = (base_pows2[i-1] * base2) % mod2
        
        len_str = [0] * n
        hash1 = [0] * n
        reverse_hash1 = [0] * n
        hash2 = [0] * n
        reverse_hash2 = [0] * n
        
        # Iterative post-order traversal
        stack = [(root, False)]
        while stack:
            node, visited = stack.pop()
            if node is None:
                continue
            if visited:
                # Compute length
                current_len = 1
                for y in children[node]:
                    current_len += len_str[y]
                len_str[node] = current_len
                
                # Compute hash1
                current_hash1 = 0
                for y in children[node]:
                    current_hash1 = (current_hash1 * base_pows1[len_str[y]] + hash1[y]) % mod1
                current_hash1 = (current_hash1 * base1 + ord(s[node])) % mod1
                hash1[node] = current_hash1
                
                # Compute reverse_hash1
                current_rev_hash1 = 0
                for y in reversed(children[node]):
                    current_rev_hash1 = (current_rev_hash1 * base_pows1[len_str[y]] + reverse_hash1[y]) % mod1
                current_rev_hash1 = (current_rev_hash1 * base1 + ord(s[node])) % mod1
                reverse_hash1[node] = current_rev_hash1
                
                # Compute hash2
                current_hash2 = 0
                for y in children[node]:
                    current_hash2 = (current_hash2 * base_pows2[len_str[y]] + hash2[y]) % mod2
                current_hash2 = (current_hash2 * base2 + ord(s[node])) % mod2
                hash2[node] = current_hash2
                
                # Compute reverse_hash2
                current_rev_hash2 = 0
                for y in reversed(children[node]):
                    current_rev_hash2 = (current_rev_hash2 * base_pows2[len_str[y]] + reverse_hash2[y]) % mod2
                current_rev_hash2 = (current_rev_hash2 * base2 + ord(s[node])) % mod2
                reverse_hash2[node] = current_rev_hash2
            else:
                stack.append((node, True))
                # Push children in reverse order to process them in order
                for y in reversed(children[node]):
                    stack.append((y, False))
        
        # Now, for each node, check if hash equals reverse_hash for both hash functions
        answer = [False] * n
        for i in range(n):
            if hash1[i] == reverse_hash1[i] and hash2[i] == reverse_hash2[i]:
                answer[i] = True
            else:
                answer[i] = False
        return answer