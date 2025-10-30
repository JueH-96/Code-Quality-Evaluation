import collections # Import is not strictly necessary as set is built-in, but good practice if used explicitly later. Can be removed.

class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        """
        Calculates the minimum possible sum of a k-avoiding array of length n.

        A k-avoiding array is an array of distinct positive integers where
        no pair of distinct elements sums to k. The goal is to construct such
        an array of length n using the smallest possible positive integers
        to minimize the total sum.

        The strategy is to greedily pick the smallest available positive integers,
        starting from 1, while ensuring the k-avoiding condition is maintained.

        Args:
            n: The desired length of the array.
            k: The target sum to avoid for pairs of distinct elements.

        Returns:
            The minimum possible sum of such an array.
        """
        
        # Use a set to efficiently store and check for numbers already added to the array.
        # This allows O(1) average time complexity for checking if a complement exists.
        seen = set()
        
        # Initialize the sum of the array elements.
        current_sum = 0
        
        # Keep track of how many elements have been added to the array.
        count = 0
        
        # Start checking positive integers from 1 upwards.
        num = 1

        # Continue the process until we have collected n elements for the array.
        while count < n:
            # Calculate the complement for the current number 'num'. 
            # If 'num' and 'complement' are both in the array, their sum is k.
            complement = k - num
            
            # Check if adding 'num' would violate the k-avoiding condition.
            # The condition is violated if 'complement' is already present in the 'seen' set.
            # We only need to check `complement not in seen` because:
            # 1. If complement > 0 and complement is in seen: We must skip num.
            # 2. If complement > 0 and complement is not in seen: We can add num.
            # 3. If complement == num (i.e., num = k/2): 'num' cannot be in 'seen' yet (as we are considering adding it), 
            #    so `complement not in seen` is true, and we can add num. This is valid because the condition applies to *distinct* elements.
            # 4. If complement <= 0: 'complement' cannot be in 'seen' because 'seen' only stores positive integers. 
            #    So `complement not in seen` is true, and we can add num.
            if complement not in seen:
                # If the condition is not violated, add 'num' to our set of chosen numbers.
                seen.add(num)
                # Add 'num' to the running total sum.
                current_sum += num
                # Increment the count of numbers successfully added to the array.
                count += 1
            # Else (if complement is in seen), we skip 'num' because adding it
            # would create a pair (num, complement) that sums to k.
            
            # Move to the next positive integer to consider for inclusion.
            num += 1
            
        # Once n elements have been collected, return the calculated minimum sum.
        return current_sum