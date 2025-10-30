class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        if n == 0:
            return []
        
        # Build the tree structure with children sorted by their indices
        children = [[] for _ in range(n)]
        for i in range(1, n):
            children[parent[i]].append(i)
        for i in range(n):
            children[i].sort()
        
        # Precompute powers of the base modulo MOD up to the maximum needed length
        base = 911382629
        MOD = 10**18 + 3
        max_pow = n + 1
        pow_p = [1] * (max_pow + 1)
        for i in range(1, max_pow + 1):
            pow_p[i] = (pow_p[i-1] * base) % MOD
        
        forward_hash = [0] * n
        reverse_hash = [0] * n
        length = [0] * n
        
        # Iterative post-order traversal using a stack
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reverse order to process them in the correct order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                # Compute forward_hash and length
                fh = 0
                ln = 0
                for child in children[node]:
                    fh = (fh * pow_p[length[child]] + forward_hash[child]) % MOD
                    ln += length[child]
                # Append current node's character
                fh = (fh * base + ord(s[node])) % MOD
                ln += 1
                forward_hash[node] = fh
                length[node] = ln
                
                # Compute reverse_hash
                rh = ord(s[node])
                rev_ln = 1
                for child in reversed(children[node]):
                    rh = (rh * pow_p[length[child]] + reverse_hash[child]) % MOD
                    rev_ln += length[child]
                reverse_hash[node] = rh
        
        # Determine the answer for each node
        answer = [forward_hash[i] == reverse_hash[i] for i in range(n)]
        return answer