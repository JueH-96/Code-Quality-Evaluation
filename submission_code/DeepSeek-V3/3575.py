from itertools import combinations

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        max_val = 0
        # Generate all possible subsequences of size 2*k
        for seq in combinations(nums, 2 * k):
            # Split the sequence into two parts
            first_half = seq[:k]
            second_half = seq[k:]
            # Compute the OR of the first half
            or_first = 0
            for num in first_half:
                or_first |= num
            # Compute the OR of the second half
            or_second = 0
            for num in second_half:
                or_second |= num
            # Compute the XOR of the two OR results
            current_val = or_first ^ or_second
            # Update the maximum value if the current value is greater
            if current_val > max_val:
                max_val = current_val
        return max_val