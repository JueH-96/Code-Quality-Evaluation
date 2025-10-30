class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def cyclic_distance(c1, c2):
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)
        
        result = list(s)
        remaining_k = k
        
        for i in range(len(s)):
            current_char = s[i]
            
            # Cost to change to 'a'
            cost_to_a = cyclic_distance(current_char, 'a')
            
            if cost_to_a <= remaining_k:
                # Change to 'a'
                result[i] = 'a'
                remaining_k -= cost_to_a
            elif remaining_k > 0:
                # Can't reach 'a', but try to get closer
                # Find the lexicographically smallest character we can reach
                best_char = current_char
                
                for target in 'abcdefghijklmnopqrstuvwxyz':
                    cost = cyclic_distance(current_char, target)
                    if cost <= remaining_k and target < best_char:
                        best_char = target
                
                if best_char != current_char:
                    cost = cyclic_distance(current_char, best_char)
                    result[i] = best_char
                    remaining_k -= cost
        
        return ''.join(result)