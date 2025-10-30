import collections
from typing import List

class Solution:
    """
    This class provides a solution to count pairs of almost equal numbers in a list.
    Two positive integers x and y are almost equal if they can become equal after performing 
    at most one swap of digits on either x or y.
    """
    
    def generate_almost_equal_set(self, num: int, cache: dict) -> set:
        """
        Generates the set of numbers that can be obtained from 'num' by performing
        at most one swap of digits. Uses a cache (memoization) to store results 
        for previously computed numbers to avoid redundant calculations.
        
        Args:
            num: The positive integer number.
            cache: A dictionary used for memoization, mapping numbers to their set of almost equal variants.
            
        Returns:
            A set of integers including 'num' (representing zero swaps) and all distinct integers 
            obtainable by performing exactly one swap of digits within 'num'.
        """
        # Check if the result for 'num' is already computed and stored in the cache.
        if num in cache:
            return cache[num]

        # Convert the number to its string representation to easily access and swap digits.
        s = str(num)
        L = len(s)
        
        # Initialize the result set. It always contains the number itself, 
        # corresponding to the case of performing zero swaps.
        result = {num}
        
        # Convert the string to a list of characters because strings are immutable in Python,
        # while lists allow element modification (needed for swapping).
        s_list = list(s)
        
        # Iterate through all distinct pairs of indices (p, q) such that 0 <= p < q < L.
        # These represent the positions of the digits to be swapped.
        for p in range(L):
            for q in range(p + 1, L):
                # Perform the swap of digits at positions p and q.
                s_list[p], s_list[q] = s_list[q], s_list[p]
                
                # Convert the modified list of characters back into a string.
                # Then convert the string representation back to an integer.
                # Python's int() function automatically handles potential leading zeros resulting from swaps
                # (e.g., swapping '3' and '0' in "30" results in "03", which int() converts to 3).
                swapped_num = int("".join(s_list))
                
                # Add the integer obtained after the swap to the result set.
                # Sets automatically handle duplicates, so adding an existing number has no effect.
                result.add(swapped_num)
                
                # Crucially, swap the digits back to their original positions (p, q) in s_list.
                # This restores the list to its state before the current swap, ensuring that
                # the next iteration swaps digits based on the original number configuration.
                s_list[p], s_list[q] = s_list[q], s_list[p] 
        
        # Store the computed set in the cache with 'num' as the key before returning.
        cache[num] = result
        # Return the completed set of numbers almost equal to 'num'.
        return result

    def countPairs(self, nums: List[int]) -> int:
        """
        Counts the number of pairs of indices (i, j) in the input list `nums` such that i < j 
        and the numbers nums[i] and nums[j] are almost equal.
        
        Args:
            nums: A list of positive integers.
            
        Returns:
            The total count of pairs (i, j) satisfying the conditions.
        """
        # Use collections.Counter to efficiently count the frequency of each number in the input list `nums`.
        # `counts` will be a dictionary mapping each number to its number of occurrences.
        counts = collections.Counter(nums)
        
        # Initialize the total count of almost equal pairs to zero.
        total_pairs = 0
        
        # Initialize a cache dictionary to store the results of `generate_almost_equal_set`.
        # This memoization avoids recomputing the set for the same number multiple times.
        cache = {} 
        
        # Initialize a set to keep track of pairs of distinct numbers {x, y} that have already been counted.
        # This is essential to prevent double counting. For example, if we count pairs related to (x, y), 
        # we don't want to count them again when processing (y, x).
        processed_pairs = set()

        # Get a list of unique numbers present in the input list `nums`. These are the keys of the `counts` dictionary.
        unique_nums = list(counts.keys())
        
        # Iterate through each unique number `x` present in `nums`.
        for x in unique_nums:
            # Retrieve the frequency count of the current number `x`.
            count_x = counts[x]
            
            # Case 1: Count pairs (i, j) where nums[i] == nums[j] == x.
            # If a number `x` appears `count_x` times, any pair of these occurrences forms an almost equal pair 
            # (since x is almost equal to x with zero swaps).
            # The number of such pairs is the number of ways to choose 2 indices from `count_x` occurrences, 
            # which is given by the combination formula C(count_x, 2) = count_x * (count_x - 1) / 2.
            if count_x > 1:
                # Use integer division // for the combination calculation.
                total_pairs += count_x * (count_x - 1) // 2
            
            # Generate the set P(x) containing `x` and all numbers obtainable from `x` by exactly one swap.
            # This function utilizes the cache for efficiency.
            P_x = self.generate_almost_equal_set(x, cache)
            
            # Iterate through each number `y` in the generated set P(x).
            for y in P_x:
                # Case 2: Count pairs (i, j) where nums[i] = x, nums[j] = y, and x != y.
                # We are interested only when `y` is different from `x`.
                # Additionally, `y` must also be present in the original `nums` list. Check this efficiently using the `counts` map.
                if y != x and y in counts:
                    # Retrieve the frequency count of `y`.
                    count_y = counts[y]
                    
                    # To ensure that we count the contribution of the pair type {x, y} only once, 
                    # we use a canonical representation for the pair. A sorted tuple `(min(x, y), max(x, y))` serves this purpose.
                    pair_key = tuple(sorted((x, y)))
                    
                    # Check if this pair type (represented by `pair_key`) has already been processed.
                    # The `processed_pairs` set stores the keys of pairs already counted.
                    if pair_key not in processed_pairs:
                        # If this pair type {x, y} hasn't been counted yet, calculate the number of pairs (i, j) formed.
                        # For each of the `count_x` occurrences of `x` and `count_y` occurrences of `y`, 
                        # we form `count_x * count_y` pairs (i, j) where nums[i]=x and nums[j]=y (or vice versa).
                        total_pairs += count_x * count_y
                        # Mark this pair type {x, y} as processed by adding its key to the set.
                        # This prevents counting pairs related to {y, x} later if y is processed.
                        processed_pairs.add(pair_key)
                        
        # After iterating through all unique numbers and considering all pair types,
        # return the final accumulated count of almost equal pairs.
        return total_pairs