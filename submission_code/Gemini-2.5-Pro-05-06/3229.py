import math

class Solution:
  def minimumCost(self, nums: list[int]) -> int:
    n = len(nums)
    nums.sort()
    
    m_val = nums[n // 2]

    s_m_val = str(m_val)
    L = len(s_m_val)
    
    candidates = set()

    # Candidate 1: Largest palindrome of length L-1
    if L > 1:
        # Forms "9", "99", "999", ...
        # Positive and < 10^9. Max L-1 is 9 (for m_val=10^9, L=10). "9"*9 is 999,999,999.
        val_l_minus_1 = int("9" * (L - 1))
        candidates.add(val_l_minus_1)
    
    # Candidate 2: Smallest palindrome of length L+1
    # Smallest palindrome of L+1 digits is "1" + "0"*(L-1) + "1".
    # Must be < 10^9, so L+1 <= 9 digits. Thus L <= 8.
    if L <= 8: 
        val_l_plus_1 = int("1" + "0" * (L - 1) + "1")
        # This is positive and < 10^9 as L+1 <= 9.
        candidates.add(val_l_plus_1)

    # Candidates 3, 4, 5: Palindromes of length L
    prefix_len = (L + 1) // 2
    prefix_s_m_val = s_m_val[0 : prefix_len]
    p_int = int(prefix_s_m_val)
    
    for delta in [-1, 0, 1]:
        current_pref_val = p_int + delta
        
        if current_pref_val <= 0: # Palindromes must be positive
            continue

        s_current_pref = str(current_pref_val)
        
        # Only consider if it forms an L-digit palindrome
        if len(s_current_pref) == prefix_len:
            if L % 2 == 1: # Odd length L
                pal_str = s_current_pref + s_current_pref[:-1][::-1]
            else: # Even length L
                pal_str = s_current_pref + s_current_pref[::-1]
            
            pal_val = int(pal_str)
            # Palindromes must be < 10^9.
            # This check is important if L=10 (m_val=10^9), as L-digit palindromes would be >=10^9.
            if pal_val < 10**9: 
                 candidates.add(pal_val)
    
    # Fallback if candidate list is empty (e.g. m_val is large and filters remove all candidates)
    # This should not happen with L-1 or L+1 candidates (e.g. for m_val=10^9, 999,999,999 is a candidate)
    # Or for small m_val (e.g. m_val=1, candidates {1,2,11} are generated)
    if not candidates:
        candidates.add(1) # Smallest positive palindrome

    p1 = -1 # Stores largest palindrome <= m_val from candidates
    p2 = -1 # Stores smallest palindrome >= m_val from candidates

    for pal_cand_val in candidates:
        if pal_cand_val <= m_val:
            if p1 == -1 or pal_cand_val > p1:
                p1 = pal_cand_val
        if pal_cand_val >= m_val:
            if p2 == -1 or pal_cand_val < p2:
                p2 = pal_cand_val
    
    min_total_cost = float('inf')

    if p1 != -1:
        current_cost_p1 = 0
        for x_num in nums:
            current_cost_p1 += abs(x_num - p1)
        min_total_cost = min(min_total_cost, current_cost_p1)

    if p2 != -1:
        current_cost_p2 = 0
        for x_num in nums:
            current_cost_p2 += abs(x_num - p2)
        min_total_cost = min(min_total_cost, current_cost_p2)
            
    return min_total_cost