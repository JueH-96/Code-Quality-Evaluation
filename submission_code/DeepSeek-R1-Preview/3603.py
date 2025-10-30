class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Build the children list for each node, sorted in increasing order
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        # Sort each children list
        for i in range(n):
            children[i].sort()
        
        # Hashing parameters
        base = 911382629
        mod = 10**18 + 3
        
        # Arrays to store fhash, bhash, and length of the subtree string for each node
        fhash = [0] * n
        bhash = [0] * n
        length = [0] * n
        answer = [False] * n
        
        # Post-order traversal using a stack
        stack = [(0, False)]
        while stack:
            node, visited = stack.pop()
            if not visited:
                stack.append((node, True))
                # Push children in reversed order to process them in the correct order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                # Process the node
                sum_children_fhash = 0
                sum_child_lengths = 0
                for child in children[node]:
                    child_len = length[child]
                    pow_term = pow(base, child_len, mod)
                    sum_children_fhash = (sum_children_fhash * pow_term + fhash[child]) % mod
                    sum_child_lengths += child_len
                
                # Compute fhash for current node
                char_val = ord(s[node]) - ord('a')
                fhash_i = (sum_children_fhash * base + char_val) % mod
                fhash[node] = fhash_i
                len_i = sum_child_lengths + 1
                length[node] = len_i
                
                # Compute sum_rev_children_bhash
                sum_rev_children_bhash = 0
                for child in reversed(children[node]):
                    child_len = length[child]
                    pow_term = pow(base, child_len, mod)
                    sum_rev_children_bhash = (sum_rev_children_bhash * pow_term + bhash[child]) % mod
                
                # Compute bhash for current node
                bhash_i = (char_val * pow(base, sum_child_lengths, mod) + sum_rev_children_bhash) % mod
                bhash[node] = bhash_i
                
                # Check if the current node's string is a palindrome
                answer[node] = (fhash_i == bhash_i)
        
        return answer