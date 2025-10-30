class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        arr = nums[:]
        ops = 0
        while True:
            non_decreasing = True
            for i in range(1, len(arr)):
                if arr[i] < arr[i-1]:
                    non_decreasing = False
                    break
            if non_decreasing:
                return ops
            min_sum = float('inf')
            min_index = -1
            for i in range(len(arr) - 1):
                s = arr[i] + arr[i+1]
                if s < min_sum:
                    min_sum = s
                    min_index = i
            arr = arr[:min_index] + [min_sum] + arr[min_index+2:]
            ops += 1