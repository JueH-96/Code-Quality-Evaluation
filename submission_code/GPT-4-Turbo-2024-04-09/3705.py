class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary to count occurrences of each number in k-sized subarrays
        count_in_subarrays = defaultdict(int)
        
        # Iterate over all possible k-sized subarrays
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            unique_elements = set(subarray)
            for element in unique_elements:
                count_in_subarrays[element] += 1
        
        # Find the largest number that appears in exactly one k-sized subarray
        largest_almost_missing = -1
        for num, count in count_in_subarrays.items():
            if count == 1:
                if num > largest_almost_missing:
                    largest_almost_missing = num
        
        return largest_almost_missing