class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        memo = {}
        
        def dp(i, mask, changed):
            """
            i: current index in string
            mask: bit mask representing characters in current partition
            changed: boolean indicating if we've used our one allowed change
            """
            if i == len(s):
                return 1  # End of string, we've completed the current partition
            
            key = (i, mask, changed)
            if key in memo:
                return memo[key]
            
            char_idx = ord(s[i]) - ord('a')
            char_bit = 1 << char_idx
            
            # Count distinct characters in current partition
            distinct_chars = bin(mask).count('1')
            
            # Option 1: Keep current character as is
            if mask & char_bit:
                # Character already in partition, continue
                result = dp(i + 1, mask, changed)
            elif distinct_chars < k:
                # Add character to partition
                result = dp(i + 1, mask | char_bit, changed)
            else:
                # Start new partition
                result = 1 + dp(i + 1, char_bit, changed)
            
            # Option 2: Change current character (if we haven't already)
            if not changed:
                for new_idx in range(26):
                    new_bit = 1 << new_idx
                    if new_bit == char_bit:  # Skip if same as original
                        continue
                    
                    if mask & new_bit:
                        # New character already in partition
                        option = dp(i + 1, mask, True)
                    elif distinct_chars < k:
                        # Add new character to partition
                        option = dp(i + 1, mask | new_bit, True)
                    else:
                        # Start new partition
                        option = 1 + dp(i + 1, new_bit, True)
                    
                    result = max(result, option)
            
            memo[key] = result
            return result
        
        return dp(0, 0, False)