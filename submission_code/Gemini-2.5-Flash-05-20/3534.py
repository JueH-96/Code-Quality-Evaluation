from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        count = 0
        n = len(nums)

        # Helper function to check if `num_str` can be transformed into `target_val`
        # by at most one swap of digits within `num_str`.
        # This function correctly handles:
        # 1. No swap needed (0 swaps) if int(num_str) is already target_val.
        # 2. Exactly one swap.
        # 3. Leading zeros created by a swap (e.g., "30" -> "03" becomes 3).
        def check_transform(num_str: str, target_val: int) -> bool:
            # Convert num_str to integer for direct comparison.
            # If it already equals target_val, no swap is needed (0 swaps).
            if int(num_str) == target_val:
                return True

            s_len = len(num_str)
            
            # Check for transformation with exactly one swap.
            # Iterate through all unique pairs of indices (i, j) to swap.
            for i in range(s_len):
                for j in range(i + 1, s_len):
                    s_list = list(num_str)
                    # Perform the swap
                    s_list[i], s_list[j] = s_list[j], s_list[i]
                    swapped_str = "".join(s_list)
                    
                    # Convert the swapped string back to an integer.
                    # int() handles leading zeros correctly (e.g., "03" becomes 3).
                    if int(swapped_str) == target_val:
                        return True
            
            # If no transformation (0 or 1 swap) leads to target_val, return False.
            return False

        # Iterate through all unique pairs (i, j) such that i < j.
        for i in range(n):
            for j in range(i + 1, n):
                x = nums[i]
                y = nums[j]

                # Two numbers x and y are almost equal if:
                # - x can be transformed into y (by 0 or 1 swap in x)
                # OR
                # - y can be transformed into x (by 0 or 1 swap in y)
                if check_transform(str(x), y) or check_transform(str(y), x):
                    count += 1
                    
        return count