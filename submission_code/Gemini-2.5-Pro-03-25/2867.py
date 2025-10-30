from typing import List

class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        """
        Calculates the number of ways to split the array nums into good subarrays.
        A good subarray contains exactly one element with the value 1.
        The result is returned modulo 10^9 + 7.

        The core idea is that any split of the array into good subarrays must correspond to partitioning the array such that each partition contains exactly one '1'.
        This means if the indices of '1's are i_1, i_2, ..., i_k, the split must be of the form S_1, S_2, ..., S_k, where S_j contains the '1' at index i_j.
        
        The split point between subarray S_j and S_{j+1} must occur somewhere after index i_j and before or at index i_{j+1}-1.
        Let the split occur after index p. Then $i_j \le p < i_{j+1}$.
        The number of choices for p is $i_{j+1} - i_j$. This is the number of ways to define the boundary between S_j and S_{j+1}.
        
        The total number of ways to split the entire array is the product of the number of choices for each boundary.
        Total ways = product(i_{j+1} - i_j) for j from 1 to k-1.

        Example: nums = [0, 1, 0, 0, 1]
        1s are at indices 1 and 4. There is one boundary to determine between the first and second '1'.
        The distance is 4 - 1 = 3.
        Number of ways = 3.
        The possible splits are:
        - [0,1] | [0,0,1]  (split after index 1)
        - [0,1,0] | [0,1]  (split after index 2)
        - [0,1,0,0] | [1]  (split after index 3)

        Example: nums = [1, 0, 1, 0, 1]
        1s are at indices 0, 2, 4. There are two boundaries.
        Boundary 1: between the '1' at index 0 and the '1' at index 2. Distance = 2 - 0 = 2.
        Boundary 2: between the '1' at index 2 and the '1' at index 4. Distance = 4 - 2 = 2.
        Total ways = 2 * 2 = 4.
        
        Edge Cases:
        - If no 1s are present (k=0), it's impossible to form any good subarray. Return 0.
        - If exactly one 1 is present (k=1), the only way is to take the entire array as one good subarray. Return 1.

        The algorithm iterates through the array once to find the indices of '1's and compute the product of distances between consecutive '1's.
        Time complexity: O(N), where N is the length of nums.
        Space complexity: O(1), as we only need to store the index of the previous '1' and the running product.
        """
        MOD = 10**9 + 7
        
        # Index of the previously found '1'. Initialized to -1 to indicate none found yet.
        prev_one_idx = -1
        # Count of '1's found in the array.
        count_ones = 0
        # Stores the product of distances between consecutive '1's. 
        # Initialized to 1 because it's a product accumulation.
        # If there's only one '1', this value (1) will be returned, which is correct.
        result = 1
        
        # Iterate through the array with index and value
        for i, num in enumerate(nums):
            if num == 1:
                # If the current element is '1'
                count_ones += 1 # Increment the count of '1's found
                
                # If this is not the first '1' encountered (i.e., prev_one_idx has been set)
                if prev_one_idx != -1:
                    # Calculate the distance between the current '1' and the previous '1'
                    diff = i - prev_one_idx
                    # The number of ways to split between these two '1's is equal to their distance.
                    # Multiply the total number of ways found so far by this distance.
                    # Apply modulo operation at each step to keep the result within bounds.
                    result = (result * diff) % MOD
                
                # Update the index of the last seen '1' to the current index
                prev_one_idx = i
            
        # After iterating through the entire array, determine the final answer based on the count of '1's.
        if count_ones == 0:
            # If no '1's were found in the array, it's impossible to form any good subarray.
            # A good subarray must contain exactly one '1'.
            return 0
        else:
            # If at least one '1' was found:
            # - If count_ones == 1, the loop for calculating product ('result = (result * diff) % MOD') 
            #   was never executed because the condition 'prev_one_idx != -1' was false when the first '1' was found. 
            #   'result' remained 1. This is correct, as there's only 1 way: the entire array is the single good subarray.
            # - If count_ones > 1, 'result' holds the product of distances modulo MOD. This is the correct number of ways.
            return result