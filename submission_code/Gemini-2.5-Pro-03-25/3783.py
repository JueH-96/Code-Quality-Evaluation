import math
from typing import List

class Solution:
    
    def __init__(self):
        """
        Initializes memoization caches for factorial and permutation counts.
        These caches are instance variables. They should be reset at the start of each call to permute.
        """
        self.memo_fact = {0: 1}
        self.memo_count = {}

    def factorial(self, k: int) -> int:
        """
        Calculates factorial of k using memoization.
        Handles non-negative integers k. Returns 0 for k < 0.
        Uses dynamic programming approach by extending from the largest computed factorial.
        """
        if k < 0: return 0
        # Check cache first
        if k in self.memo_fact: return self.memo_fact[k]
        
        # Ensure base case 0! = 1 is present (should be by initialization)
        if not self.memo_fact: self.memo_fact[0] = 1
        
        # Find the largest key already computed
        max_computed = max(self.memo_fact.keys())
        res = self.memo_fact[max_computed]
        
        # Compute factorials incrementally from max_computed up to k
        for i in range(max_computed + 1, k + 1):
             res *= i
             # Store intermediate results in cache
             self.memo_fact[i] = res
        return res

    def get_count(self, num_odd: int, num_even: int, first_is_odd: bool) -> int:
        """ 
        Calculates the number of alternating permutations possible using a specific set of 
        num_odd odd numbers and num_even even numbers, where the parity 
        of the first element is specified by first_is_odd.
        Uses memoization to store and retrieve results for state (num_odd, num_even, first_is_odd).
        """
        # Check state validity and memoization cache
        state = (num_odd, num_even, first_is_odd)
        if state in self.memo_count: return self.memo_count[state]

        # Invalid counts return 0 permutations
        if num_odd < 0 or num_even < 0: return 0
        
        # Base case: If no numbers left, there's exactly one way (the empty permutation suffix)
        if num_odd == 0 and num_even == 0: return 1 

        res = 0
        # Calculate required factorials using the memoized factorial function
        fact_odd = self.factorial(num_odd)
        fact_even = self.factorial(num_even)

        # Check if the requested structure is possible given the counts of odd/even numbers
        if first_is_odd:
            # Required pattern starts OEOE... 
            # This requires num_odd >= num_even and the difference is at most 1.
            if num_odd == num_even or num_odd == num_even + 1:
                 # The number of ways is arranging odd numbers in odd slots (num_odd!)
                 # and even numbers in even slots (num_even!)
                 res = fact_odd * fact_even
        else: # first element must be even
            # Required pattern starts EOEO...
            # This requires num_even >= num_odd and the difference is at most 1.
            if num_even == num_odd or num_even == num_odd + 1:
                 # The number of ways is arranging odd numbers in odd slots (num_odd!)
                 # and even numbers in even slots (num_even!)
                 res = fact_odd * fact_even
        
        # Store result in cache before returning
        self.memo_count[state] = res
        return res

    def permute(self, n: int, k: int) -> List[int]:
        """
        Finds the k-th lexicographically smallest alternating permutation of numbers 1 to n.
        An alternating permutation is one where adjacent elements have different parity.
        Returns an empty list if k is out of range (less than 1 or greater than total count).
        """
        
        # Reset memoization caches for this new call. This ensures state isolation between calls.
        self.memo_fact = {0: 1}
        self.memo_count = {}

        # Calculate initial counts of odd and even numbers in the range [1, n].
        N_odd = (n + 1) // 2 # Equivalent to math.ceil(n / 2)
        N_even = n // 2     # Equivalent to math.floor(n / 2)

        # Calculate the total number of alternating permutations possible for n.
        # Sum counts of permutations starting with odd and permutations starting with even.
        total_count_odd_start = self.get_count(N_odd, N_even, True)
        total_count_even_start = self.get_count(N_odd, N_even, False)
        total_count = total_count_odd_start + total_count_even_start

        # Check if k is valid (1 <= k <= total_count).
        if k <= 0 or k > total_count: 
            return []

        result = []
        # Maintain sorted lists of available odd and even numbers. List allows efficient pop from index.
        available_odd = list(range(1, n + 1, 2))
        available_even = list(range(2, n + 1, 2))
        num_odd = N_odd
        num_even = N_even
        
        current_k = k # Use a working copy of k, which will be decremented.

        # Construct the permutation element by element from left to right (index i from 1 to n).
        for i in range(1, n + 1):
            last_p_parity = -1 # Parity of the previously chosen element. -1 for initial state.
            if i > 1:
                last_p_parity = result[-1] % 2 # 0 for even, 1 for odd

            candidate_found = False
            # Indices to iterate through the available odd and even lists.
            current_odd_idx = 0
            current_even_idx = 0
            
            # Iterate through candidates in lexicographical order.
            # This requires considering the minimum available number across both lists at each step.
            while current_odd_idx < len(available_odd) or current_even_idx < len(available_even):
                
                # Determine the globally smallest available number (candidate).
                candidate = -1
                is_odd = False # Flag indicating if the candidate is odd
                
                # Safely get values; use infinity if a list is exhausted to facilitate comparison.
                odd_val = available_odd[current_odd_idx] if current_odd_idx < len(available_odd) else float('inf')
                even_val = available_even[current_even_idx] if current_even_idx < len(available_even) else float('inf')

                # Select the smaller value as the current candidate.
                if odd_val < even_val:
                    candidate = odd_val
                    is_odd = True
                else:
                    candidate = even_val
                    is_odd = False

                # Check if the candidate's parity satisfies the alternating constraint.
                valid_parity = False
                if i == 1: # First element can be anything (provided a valid sequence exists).
                    valid_parity = True
                else: 
                    current_parity = candidate % 2
                    # Must have different parity than the previous element.
                    if current_parity != last_p_parity:
                        valid_parity = True

                if valid_parity:
                    # If parity is valid, calculate the count of permutations that start with the current prefix + candidate.
                    count_suffix = 0
                    if is_odd:
                        # If candidate is odd, the next element must be even.
                        count_suffix = self.get_count(num_odd - 1, num_even, False)
                    else: # candidate is even
                        # If candidate is even, the next element must be odd.
                        count_suffix = self.get_count(num_odd, num_even - 1, True)

                    # Check if the k-th permutation falls within the block defined by this candidate.
                    if current_k <= count_suffix:
                        # Yes, this candidate is the correct choice for position i.
                        result.append(candidate)
                        # Remove the chosen candidate from the appropriate available list.
                        if is_odd:
                            available_odd.pop(current_odd_idx) # List pop by index is O(N-idx)
                            num_odd -= 1
                        else:
                            available_even.pop(current_even_idx) # List pop by index is O(N-idx)
                            num_even -= 1
                        candidate_found = True
                        # Break from the while loop finding candidates for position i, move to next position i+1.
                        break 
                    else:
                        # No, the k-th permutation is later. Reduce k by the count of this block.
                        current_k -= count_suffix
                
                # Advance the index for the list from which the candidate was considered.
                # This moves to the next candidate in lexicographical order.
                if is_odd:
                    current_odd_idx += 1
                else:
                    current_even_idx += 1

            # Safety check: If no candidate was found for position i, it indicates an issue.
            # This should not happen if k was initially valid and calculations are correct.
            if not candidate_found:
                 return [] # Return empty list indicates an unexpected error state.

        return result