class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        count = {}
        
        # Count how many subarrays each number appears in
        for i in range(len(nums) - k + 1):
            for num in set(nums[i:i+k]):
                count[num] = count.get(num, 0) + 1
        
        # Find the largest number that appears in exactly one subarray
        return max((num for num, freq in count.items() if freq == 1), default=-1)