class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s: str, can_change: bool) -> int:
            if not s:
                return 0
                
            n = len(s)
            distinct = set()
            max_len = 0
            
            # Try all possible lengths for first partition
            for i in range(n):
                distinct.add(s[i])
                if len(distinct) > k:
                    break
                max_len = i + 1
                    
            # If we can change a character
            if can_change:
                best = 1
                
                # Try changing each position in current window
                for change_pos in range(max_len):
                    orig_char = s[change_pos]
                    
                    # Try each possible letter
                    for new_char in range(ord('a'), ord('z') + 1):
                        new_char = chr(new_char)
                        if new_char == orig_char:
                            continue
                            
                        # Create new string with changed character
                        new_s = s[:change_pos] + new_char + s[change_pos + 1:]
                        
                        # Count distinct in new window
                        window = set(new_s[:max_len])
                        if len(window) <= k:
                            best = max(best, 1 + count_partitions(new_s[max_len:], False))
                            
                # Also try not changing anything in current window
                best = max(best, 1 + count_partitions(s[max_len:], True))
                return best
                
            # No more changes allowed
            return 1 + count_partitions(s[max_len:], False)
            
        return count_partitions(s, True)