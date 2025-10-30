from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        count = 0
        current_arr = nums[:]
        while True:
            if self.is_non_decreasing(current_arr):
                return count
            # Find the leftmost minimum sum adjacent pair
            min_sum = float('inf')
            pos = 0
            for i in range(len(current_arr) - 1):
                current_sum = current_arr[i] + current_arr[i+1]
                if current_sum < min_sum:
                    min_sum = current_sum
                    pos = i
            # Merge the pair at position 'pos'
            current_arr = current_arr[:pos] + [current_arr[pos] + current_arr[pos+1]] + current_arr[pos+2:]
            count += 1
    
    def is_non_decreasing(self, arr: List[int]) -> bool:
        for i in range(len(arr) - 1):
            if arr[i] > arr[i+1]:
                return False
        return True