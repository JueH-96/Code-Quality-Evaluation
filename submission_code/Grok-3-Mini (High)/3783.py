import typing
from typing import List

class Solution:
    def permute(self, n: int, k: int) -> List[int]:
        # Precompute factorial up to 100
        fact = [1]
        for i in range(1, 101):
            fact.append(fact[-1] * i)
        
        # Calculate number of odd and even numbers
        num_odd = (n + 1) // 2
        num_even = n // 2
        
        # Calculate total number of alternating permutations
        if n % 2 == 1:  # n is odd
            total = fact[num_odd] * fact[num_even]
            start_req_parity = 'odd'
        else:  # n is even
            total = 2 * fact[num_even] * fact[num_even]
            start_req_parity = 'any'
        
        # If k is larger than total, return empty list
        if k > total:
            return []
        
        # Initialize permutation and used array
        perm = [0] * n
        used = [False] * (n + 1)  # Index 0 is unused, 1 to n are for numbers
        
        # Define helper function
        def helper(pos, rem_odd, rem_even, req_parity, k):
            if pos == n:
                return  # Permutation is complete
            
            # Compute S based on required parity
            if req_parity == 'odd':
                S = fact[rem_odd - 1] * fact[rem_even]
                candidates = [num for num in range(1, n + 1) if not used[num] and num % 2 == 1]
            elif req_parity == 'even':
                S = fact[rem_odd] * fact[rem_even - 1]
                candidates = [num for num in range(1, n + 1) if not used[num] and num % 2 == 0]
            elif req_parity == 'any':
                S = fact[rem_even] * fact[rem_odd - 1]  # rem_odd and rem_even are equal when 'any'
                candidates = [num for num in range(1, n + 1) if not used[num]]
            
            # Candidates list is sorted since range is in order
            C = len(candidates)
            idx = (k - 1) // S  # 0-based index
            chosen_num = candidates[idx]
            new_k = k - (idx * S)  # New k for the subtree
            
            # Set the chosen number in permutation and mark as used
            perm[pos] = chosen_num
            used[chosen_num] = True
            
            # Compute new remaining counts and next required parity
            if chosen_num % 2 == 1:  # Odd
                new_rem_odd = rem_odd - 1
                new_rem_even = rem_even
                next_req_parity = 'even'
            else:  # Even
                new_rem_odd = rem_odd
                new_rem_even = rem_even - 1
                next_req_parity = 'odd'
            
            # Recurse to the next position
            helper(pos + 1, new_rem_odd, new_rem_even, next_req_parity, new_k)
        
        # Call the helper function starting from position 0
        helper(0, num_odd, num_even, start_req_parity, k)
        
        return perm