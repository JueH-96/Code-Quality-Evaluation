class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)

        # Helper function to check if it's possible to achieve max_len with numOps
        def can(max_len: int) -> bool:
            ops_needed = 0
            current_len_0 = 0  # Current consecutive count of '0's
            current_len_1 = 0  # Current consecutive count of '1's

            for char_idx in range(n):
                if s[char_idx] == '0':
                    current_len_0 += 1
                    current_len_1 = 0  # Streak of '1's is broken
                else: # s[char_idx] == '1'
                    current_len_1 += 1
                    current_len_0 = 0  # Streak of '0's is broken

                # Check if current streak exceeds max_len and needs a flip
                if current_len_0 > max_len:
                    ops_needed += 1
                    # If we flip s[char_idx] from '0' to '1':
                    # The '0' streak is broken.
                    current_len_0 = 0 
                    # The new character at s[char_idx] is '1', starting a new '1' streak.
                    current_len_1 = 1 
                elif current_len_1 > max_len:
                    ops_needed += 1
                    # If we flip s[char_idx] from '1' to '0':
                    # The '1' streak is broken.
                    current_len_1 = 0
                    # The new character at s[char_idx] is '0', starting a new '0' streak.
                    current_len_0 = 1
            
            return ops_needed <= numOps

        # Binary search for the minimum possible max_len
        # The answer can range from 1 to n
        low = 1
        high = n
        min_longest_substring_length = n # Initialize with max possible value

        while low <= high:
            mid = low + (high - low) // 2
            if can(mid):
                min_longest_substring_length = mid
                high = mid - 1  # Try to achieve an even smaller max_len
            else:
                low = mid + 1   # mid is too small, need a larger max_len

        return min_longest_substring_length