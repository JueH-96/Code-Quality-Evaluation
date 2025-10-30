class Solution:
    def maximumLength(self, s: str) -> int:
        n = len(s)
        max_len = -1
        
        # For each possible length
        for length in range(1, n + 1):
            # For each starting position
            count_dict = {}
            
            for start in range(n - length + 1):
                substr = s[start:start + length]
                # Check if substring is special (all same chars)
                if len(set(substr)) == 1:
                    count_dict[substr] = count_dict.get(substr, 0) + 1
                    
                    # If we found a special substring occurring at least thrice
                    if count_dict[substr] >= 3:
                        max_len = length
                        
            # If no special substring of current length occurs thrice
            # No need to check longer lengths
            if max_len != length:
                break
                
        return max_len