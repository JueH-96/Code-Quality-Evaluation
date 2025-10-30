class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        
        # Sort children for each node
        for i in range(n):
            children[i].sort()
        
        MOD = 10**9 + 7
        BASE = 911382629
        
        max_n = n + 1
        pow_base = [1] * max_n
        for i in range(1, max_n):
            pow_base[i] = (pow_base[i-1] * BASE) % MOD
        
        forward = [0] * n
        reverse_hash = [0] * n
        lens = [0] * n
        answer = [False] * n
        
        stack = [(0, False)]
        
        while stack:
            node, processed = stack.pop()
            if not processed:
                stack.append((node, True))
                # Push children in reversed order to process in sorted order
                for child in reversed(children[node]):
                    stack.append((child, False))
            else:
                ch_list = children[node]
                # Compute combined forward hash and length
                combined_f_hash = 0
                combined_f_len = 0
                for child in ch_list:
                    combined_f_hash = (combined_f_hash * pow_base[lens[child]] + forward[child]) % MOD
                    combined_f_len += lens[child]
                
                # Compute combined reverse hash and length
                combined_r_hash = 0
                combined_r_len = 0
                for child in reversed(ch_list):
                    combined_r_hash = (combined_r_hash + reverse_hash[child] * pow_base[combined_r_len]) % MOD
                    combined_r_len += lens[child]
                
                # Current node's character value
                val = ord(s[node]) - ord('a') + 1
                
                # Calculate current forward and reverse hash
                current_forward = (combined_f_hash * pow_base[1] + val) % MOD
                current_reverse = (val * pow_base[combined_r_len] + combined_r_hash) % MOD
                
                forward[node] = current_forward
                reverse_hash[node] = current_reverse
                lens[node] = combined_f_len + 1
                
                answer[node] = (current_forward == current_reverse)
        
        return answer