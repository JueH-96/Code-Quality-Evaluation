import math
from typing import List

class Solution:
  def permute(self, n: int, k: int) -> List[int]:
    # Precompute factorials
    fact = [1] * (n + 1) # fact[0] is 1
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    # Alternatively: fact = [math.factorial(i) for i in range(n + 1)]

    num_odd_total = (n + 1) // 2  # ceil(n/2)
    num_even_total = n // 2      # floor(n/2)

    memo_count_ways = {}
    def count_ways(co, ce, is_first_pos_odd):
        # co: count of ODD numbers available for the sub-permutation
        # ce: count of EVEN numbers available for the sub-permutation
        # is_first_pos_odd: True if the first slot of this sub-permutation must be ODD
        
        state = (co, ce, is_first_pos_odd)
        if state in memo_count_ways:
            return memo_count_ways[state]
        
        if co < 0 or ce < 0: # Invalid state
            return 0
        if co == 0 and ce == 0: # Base case: no more numbers to place
            return 1

        res = 0
        if is_first_pos_odd:
            if co == 0: # Cannot pick an odd if no odds are available
                res = 0
            # To form an alternating sequence of length (co+ce) starting with ODD (OEOE...):
            # Must have (co == ce) or (co == ce + 1)
            elif not (co == ce or co == ce + 1):
                res = 0
            else:
                # Ways to arrange 'co' specific odd numbers in their slots *
                # Ways to arrange 'ce' specific even numbers in their slots.
                res = fact[co] * fact[ce]
        else: # is_first_pos_even
            if ce == 0: # Cannot pick an even if no evens are available
                res = 0
            # To form an alternating sequence of length (co+ce) starting with EVEN (EOEO...):
            # Must have (co == ce) or (ce == co + 1)
            elif not (co == ce or ce == co + 1):
                res = 0
            else:
                res = fact[co] * fact[ce]
        
        memo_count_ways[state] = res
        return res

    # Calculate total number of valid alternating permutations of length n
    total_perms = 0
    # Case 1: Permutation starts with an ODD number
    total_perms += count_ways(num_odd_total, num_even_total, True) 
    
    # Case 2: Permutation starts with an EVEN number
    total_perms += count_ways(num_odd_total, num_even_total, False)
    
    if k > total_perms:
        return []

    ans = []
    used = [False] * (n + 1) # used[x] is True if number x is already in ans
    current_odd_remaining = num_odd_total
    current_even_remaining = num_even_total
    
    # is_prev_odd: True if the previously placed number (ans[-1]) was odd.
    #              False if it was even.
    #              None if ans is empty (i.e., for the first element).
    is_prev_odd = None 

    for i in range(n): # For each position ans[i] in the permutation (0 to n-1)
        
        # Determine required parity for current position ans[i] based on ans[i-1]
        required_parity_is_odd_for_current_pos = None
        if is_prev_odd is True: # Previous was odd, current must be even
            required_parity_is_odd_for_current_pos = False
        elif is_prev_odd is False: # Previous was even, current must be odd
            required_parity_is_odd_for_current_pos = True
        # If is_prev_odd is None (i=0), no restriction from previous element.
            
        for x in range(1, n + 1): # Try candidate numbers 1 to n (for lexicographical order)
            if used[x]:
                continue

            is_x_odd = (x % 2 == 1)

            # Check if x's parity matches requirement for this position (if any)
            if required_parity_is_odd_for_current_pos is not None and \
               is_x_odd != required_parity_is_odd_for_current_pos:
                continue
            
            # If x is chosen for current position ans[i]:
            # Calculate how many valid alternating permutations can be formed starting with ans + [x].
            # This count is for the suffix of the permutation.
            
            perms_count_for_this_choice = 0
            if is_x_odd:
                # If x (odd) is placed, (current_odd_remaining - 1) odds and
                # (current_even_remaining) evens are left for the suffix.
                # The NEXT slot (ans[i+1]) must be filled by an EVEN number.
                perms_count_for_this_choice = count_ways(current_odd_remaining - 1, 
                                                         current_even_remaining, 
                                                         False) # False means next slot in suffix must be even
            else: # x is even
                # If x (even) is placed, (current_odd_remaining) odds and
                # (current_even_remaining - 1) evens are left for the suffix.
                # The NEXT slot (ans[i+1]) must be filled by an ODD number.
                perms_count_for_this_choice = count_ways(current_odd_remaining, 
                                                         current_even_remaining - 1, 
                                                         True) # True means next slot in suffix must be odd
            
            if k <= perms_count_for_this_choice:
                ans.append(x)
                used[x] = True
                is_prev_odd = is_x_odd # Update for next iteration of outer loop (for ans[i+1])
                if is_x_odd:
                    current_odd_remaining -= 1
                else:
                    current_even_remaining -= 1
                break # Found number for current position i, move to (i+1)-th position
            else:
                k -= perms_count_for_this_choice # Skip these permutations, adjust k
        
    return ans