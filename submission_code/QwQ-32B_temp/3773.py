from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        count = 0
        arr = list(nums)
        while True:
            # Check if the array is non-decreasing
            is_ok = True
            for i in range(len(arr) - 1):
                if arr[i] > arr[i + 1]:
                    is_ok = False
                    break
            if is_ok:
                return count
            
            # Find the pair with the minimal sum (leftmost if ties)
            min_sum = float('inf')
            best_i = 0
            for i in range(len(arr) - 1):
                current_sum = arr[i] + arr[i + 1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    best_i = i
                # No action needed if equal since we take the first occurrence
            
            # Merge the selected pair
            new_arr = arr[:best_i] + [arr[best_i] + arr[best_i + 1]] + arr[best_i + 2:]
            arr = new_arr
            count += 1