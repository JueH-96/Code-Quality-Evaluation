import collections
from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        
        # Helper function to generate all numbers obtainable by at most one swap.
        def generate_swaps(n: int) -> set[int]:
            s = str(n)
            d = len(s)
            swaps = {n}  # Include the original number for the 0-swap case
            s_list = list(s)
            for i in range(d):
                for j in range(i + 1, d):
                    temp_list = s_list[:]
                    temp_list[i], temp_list[j] = temp_list[j], temp_list[i]
                    # Note: int() correctly handles leading zeros, e.g., int("03") is 3.
                    new_s = "".join(temp_list)
                    swaps.add(int(new_s))
            return swaps

        # Count frequencies of each number to handle duplicates efficiently.
        counts = collections.Counter(nums)
        unique_nums = list(counts.keys())
        n_unique = len(unique_nums)
        
        total_pairs = 0
        
        # Pre-compute swap sets for each unique number for efficiency.
        swap_sets = {num: generate_swaps(num) for num in unique_nums}
        
        for i in range(n_unique):
            x = unique_nums[i]
            count_x = counts[x]
            
            # Case 1: Pairs of identical numbers (e.g., two 12s).
            # If a number appears k > 1 times, these form kC2 pairs.
            if count_x > 1:
                total_pairs += count_x * (count_x - 1) // 2
            
            # Case 2: Pairs of different numbers (e.g., 12 and 21).
            # We iterate j from i + 1 to consider each pair of unique numbers once.
            for j in range(i + 1, n_unique):
                y = unique_nums[j]
                count_y = counts[y]
                
                # Check if x and y are almost equal using the pre-computed sets.
                if y in swap_sets[x] or x in swap_sets[y]:
                    # Each instance of x can be paired with each instance of y.
                    total_pairs += count_x * count_y
                    
        return total_pairs