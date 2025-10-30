class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        # Count how many subarrays of size k each integer appears in
        count = {}
        
        # Generate all subarrays of size k
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            # For each unique element in this subarray
            for num in set(subarray):
                if num not in count:
                    count[num] = 0
                count[num] += 1
        
        # Find the largest integer that appears in exactly one subarray
        result = -1
        for num, cnt in count.items():
            if cnt == 1:
                result = max(result, num)
        
        return result