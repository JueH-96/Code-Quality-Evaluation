from typing import List
import bisect

# Precompute all palindromes less than 10^9 globally.
# This ensures it's done only once, even if the Solution class is instantiated multiple times.

_PALINDROMES_SET = set()

# Iterate through numbers whose string representation forms the first half of a palindrome.
# For example, if i = 123:
#   - Odd length palindrome: '123' + reverse('12') = '12321'
#   - Even length palindrome: '123' + reverse('123') = '123321'
# The maximum i (99999) generates the largest 9-digit odd palindrome (999999999).
# Numbers larger than 9999 for 'i' will result in even palindromes > 10^9,
# so the `p_even < 10**9` check filters them out automatically.
for i in range(1, 100000): 
    s = str(i)
    
    # Generate odd length palindromes
    p_odd_str = s + s[:-1][::-1]
    p_odd = int(p_odd_str)
    if p_odd < 10**9:
        _PALINDROMES_SET.add(p_odd)

    # Generate even length palindromes
    p_even_str = s + s[::-1]
    p_even = int(p_even_str)
    if p_even < 10**9:
        _PALINDROMES_SET.add(p_even)

# Convert the set to a sorted list for efficient binary search.
_PALINDROMES_LIST = sorted(list(_PALINDROMES_SET))

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Sort the input array to find its median.
        nums.sort()
        
        # The median of the array. This value minimizes sum(|num - y|) over all real y.
        # We need to find a palindromic y close to this median.
        median_val = nums[n // 2]
        
        candidate_palindromes = []
        
        # Use binary search to find the insertion point for `median_val` in the sorted palindromes list.
        # `idx` will be the index of the first palindrome that is greater than or equal to `median_val`.
        idx = bisect.bisect_left(_PALINDROMES_LIST, median_val)
        
        # Add candidate palindromes based on the median:
        # 1. The smallest palindrome greater than or equal to `median_val`.
        if idx < len(_PALINDROMES_LIST):
            candidate_palindromes.append(_PALINDROMES_LIST[idx])
            
        # 2. The largest palindrome strictly less than `median_val`.
        # This is the palindrome just before the one found by `bisect_left`.
        if idx > 0:
            candidate_palindromes.append(_PALINDROMES_LIST[idx - 1])
        
        # Remove any potential duplicates (though with the logic above, there typically won't be if idx is properly used).
        # This step is mostly for robustness or if a wider search range was chosen.
        candidate_palindromes = list(set(candidate_palindromes))
        
        min_total_cost = float('inf')
        
        # Calculate the total cost for each candidate palindrome and find the minimum.
        for y_candidate in candidate_palindromes:
            current_cost = 0
            for num in nums:
                current_cost += abs(num - y_candidate)
            min_total_cost = min(min_total_cost, current_cost)
            
        return min_total_cost