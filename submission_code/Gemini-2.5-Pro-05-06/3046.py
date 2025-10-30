import math

class Solution:
  def minimumOperations(self, num: str) -> int:
    n = len(num)
    
    # Initialize current_min_ops with the cost of making the number "0".
    # If '0' is in num, cost to make "0" is n-1 operations
    # (delete all digits except one '0'). If num itself is "0" (n=1), then n-1 = 0 operations.
    # If '0' is not in num, cost to make "0" is n operations (delete all digits).
    # Example: num="5" (n=1), '0' not in num, current_min_ops = 1.
    if any(digit == '0' for digit in num):
        current_min_ops = n - 1
    else:
        # No '0' in num, so to make "0", all n digits must be deleted.
        current_min_ops = n
        
    # Helper function to calculate deletions for a target suffix s1_target + s2_target
    def get_deletions_for_suffix(s1_target: str, s2_target: str) -> float:
        # Find the rightmost occurrence of s2_target
        idx_s2 = num.rfind(s2_target)
        
        if idx_s2 == -1: # s2_target not found
            return math.inf
        
        # Find the rightmost occurrence of s1_target in num[0 ... idx_s2-1]
        # str.rfind(substring, start_inclusive, end_exclusive_search_range)
        # So, search in num[0 : idx_s2] (i.e., num up to, but not including, index idx_s2)
        idx_s1 = num.rfind(s1_target, 0, idx_s2) 
        
        if idx_s1 == -1: # s1_target not found before s2_target
            return math.inf
        
        # Deletions calculation:
        # 1. Characters between num[idx_s1] and num[idx_s2] must be deleted.
        #    These are at indices (idx_s1 + 1) through (idx_s2 - 1).
        #    Count = (idx_s2 - 1) - (idx_s1 + 1) + 1 = idx_s2 - idx_s1 - 1
        deletions_between = (idx_s2 - idx_s1 - 1)
        
        # 2. Characters after num[idx_s2] must be deleted.
        #    These are at indices (idx_s2 + 1) through (n - 1).
        #    Count = (n - 1) - (idx_s2 + 1) + 1 = n - 1 - idx_s2
        #    (n - 1) is the last valid index in num.
        deletions_after = (n - 1) - idx_s2
        
        return float(deletions_between + deletions_after)

    # List of target suffixes (s1, s2)
    suffixes_to_check = [("0", "0"), ("2", "5"), ("5", "0"), ("7", "5")]
    
    for s1, s2 in suffixes_to_check:
        deletions = get_deletions_for_suffix(s1, s2)
        current_min_ops = min(current_min_ops, deletions)
        
    return int(current_min_ops) # Result must be an int