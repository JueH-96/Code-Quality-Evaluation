import sys
from functools import lru_cache

# It's good practice to increase recursion limit if max depth is close to default.
# sys.setrecursionlimit(1200) # Max L=800. Python default is often 1000 or 3000.

class Solution:
  def countKReducibleNumbers(self, s: str, k: int) -> int:
    MOD = 10**9 + 7
    L = len(s)

    # memo_get_f for caching results of get_f(val)
    # This dictionary will be local to each call of countKReducibleNumbers
    memo_get_f = {}
    def get_f(val: int) -> int:
        if val == 1:
            return 0
        if val in memo_get_f:
            return memo_get_f[val]
        
        popc = bin(val).count('1') # Popcount of val
        
        # Recursion: f(val) = 1 + f(popcount(val))
        # This recursion is guaranteed to terminate because popc < val for val > 1 (except popc(2)=1<2),
        # eventually reaching val=1.
        res = 1 + get_f(popc)
        memo_get_f[val] = res
        return res

    # Optimization for k >= 5
    # The maximum value of f(c) for c >= 1 is 4.
    # (e.g., f(127) = 4, as 127 -> 7 -> 3 -> 2 -> 1).
    # If k-1 >= 4 (i.e., k >= 5), then f(c) <= k-1 is true for all c >= 1.
    # Thus, all positive integers x (with popcount c > 0) are candidates.
    # The problem asks for x in [1, n-1]. The count of such numbers is n-1.
    if k >= 5: # Given k <= 5, this implies k == 5.
        if s == "1": # n=1. Range [1, n-1] is empty.
            return 0
        
        # Calculate (n-1) % MOD
        val_N = 0
        for digit_char in s:
            val_N = (val_N * 2 + int(digit_char)) % MOD
        return (val_N - 1 + MOD) % MOD

    # Populate target_popcounts set
    # These are popcount values 'c' such that f(c) <= k-1.
    target_popcounts = set()
    # Max popcount for a number of length L is L itself (e.g. string of L ones).
    for i in range(1, L + 1): 
        if get_f(i) <= k - 1:
            target_popcounts.add(i)
    
    # If no popcount value c satisfies f(c) <= k-1.
    # This should not happen for k >= 1, because f(1)=0, so k-1 >= 0 means c=1 is always a target popcount.
    if not target_popcounts:
        return 0
    
    # Digit DP to count numbers x < n with popcount(x) in target_popcounts.
    # dp_solve(idx, current_popcount, is_less)
    # idx: current bit position being decided (0 to L-1).
    # current_popcount: popcount of the prefix s[0...idx-1].
    # is_less: boolean, True if number formed so far is strictly less than s's prefix.
    @lru_cache(None) # Memoization for DP states
    def dp_solve(idx: int, current_popcount: int, is_less: bool) -> int:
        if idx == L: # All L digits have been placed
            if not is_less: 
                # If not is_less, the number formed is n itself. We need strictly < n.
                return 0
            
            # Number formed is < n. Check if it's positive and has a target popcount.
            if current_popcount == 0: # Number 0, not positive.
                return 0
            return 1 if current_popcount in target_popcounts else 0
        
        ans = 0
        # Determine upper bound for the current digit choice.
        # If is_less is True, we can place 0 or 1 (upper_bound is 1).
        # If is_less is False, digit is limited by s[idx] (upper_bound is int(s[idx])).
        upper_bound_for_digit = 1 if is_less else int(s[idx])
        
        for digit in range(upper_bound_for_digit + 1):
            # Determine the 'is_less' status for the next recursive call.
            # If current 'is_less' is True, 'new_is_less' remains True.
            # If current 'is_less' is False:
            #   - if 'digit' < s[idx] (i.e. digit < upper_bound_for_digit), 'new_is_less' becomes True.
            #   - if 'digit' == s[idx] (i.e. digit == upper_bound_for_digit), 'new_is_less' remains False.
            # This logic is captured by: new_is_less = is_less or (digit < upper_bound_for_digit)
            new_is_less_val = is_less or (digit < upper_bound_for_digit)
            
            # Optimization: if current_popcount + digit > max_val_in_target_popcounts + (L - 1 - idx), can prune.
            # Max possible popcount is L. current_popcount can't exceed L.
            # No specific pruning here other than what results from target_popcounts set.
            if current_popcount + digit <= L : # current popcount cannot exceed L
                 ans = (ans + dp_solve(idx + 1, current_popcount + digit, new_is_less_val)) % MOD
            # If current_popcount + digit > L, then this path will lead to a popcount > L.
            # Since all target_popcounts are <= L, such a number won't be counted.
            # The check `current_popcount + digit <= L` can be seen as a small optimization.
            # If it's removed, it's still correct as non-target popcounts return 0.
            # Let's keep it to make it explicit that popcount can't grow too large.
            # However, a more precise check current_popcount + digit + (L - (idx+1)) < min_target_popcount etc.
            # would be more effective for pruning, but adds complexity.
            # The state current_popcount can go up to L. That's fine.

        return ans

    # Initial call to dp_solve. Counts numbers in [0, n-1].
    # Base case handles filtering for positive numbers and target popcounts.
    result = dp_solve(0, 0, False)
    
    # Clear caches if the Solution object or its members were persistent across test cases.
    # For LeetCode's typical setup (new object per test), this is not strictly necessary.
    # dp_solve.cache_clear() 
    # memo_get_f.clear() # memo_get_f is a local dict, so it's fresh anyway.

    return result