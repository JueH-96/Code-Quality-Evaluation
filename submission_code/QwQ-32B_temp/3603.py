class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        if n == 0:
            return []
        
        # Build children lists and sort them
        children = [[] for _ in range(n)]
        for y in range(n):
            p = parent[y]
            if p != -1:
                children[p].append(y)
        for ch in children:
            ch.sort()  # Ensure children are processed in increasing order
        
        # Precompute powers of the base modulo mod
        base = 911382629
        mod = 10**18 + 3
        max_len = n
        pow_base = [1] * (max_len + 1)
        for i in range(1, max_len + 1):
            pow_base[i] = (pow_base[i-1] * base) % mod
        
        # Arrays to hold forward, reverse hash and length
        forward = [0] * n
        reverse = [0] * n
        length = [0] * n
        
        # Iterative post-order traversal using stack
        stack = []
        stack.append((0, False))
        
        while stack:
            node, processed = stack.pop()
            if processed:
                # Compute forward, reverse, and length for this node
                forward_children = 0
                reverse_children = 0
                total_length = 0
                
                # Compute forward_children and total_length
                for y in children[node]:
                    forward_children = (forward_children * pow_base[length[y]] + forward[y]) % mod
                    total_length += length[y]
                
                # Compute forward hash
                current_char = ord(s[node])
                forward[node] = (forward_children * base + current_char) % mod
                
                # Compute reverse_children
                reverse_children = 0
                for y in reversed(children[node]):
                    reverse_children = (reverse_children * pow_base[length[y]] + reverse[y]) % mod
                
                # Compute reverse hash
                reverse[node] = (current_char * pow_base[total_length] + reverse_children) % mod
                length[node] = total_length + 1
            else:
                # Push back with processed=True and then children in reverse order
                stack.append((node, True))
                # Push children in reversed order so they are processed in original order
                for y in reversed(children[node]):
                    stack.append((y, False))
        
        # Determine the answer for each node
        answer = [ (forward[i] == reverse[i]) for i in range(n) ]
        return answer