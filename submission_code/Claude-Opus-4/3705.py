class Solution:
    def largestInteger(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Count occurrences of each number in each subarray of size k
        count = defaultdict(int)
        
        # Process all subarrays of size k
        for i in range(len(nums) - k + 1):
            # Get unique elements in current subarray
            subarray = nums[i:i+k]
            unique_in_subarray = set(subarray)
            
            # Increment count for each unique element in this subarray
            for num in unique_in_subarray:
                count[num] += 1
        
        # Find the largest number that appears in exactly one subarray
        result = -1
        for num, cnt in count.items():
            if cnt == 1:
                result = max(result, num)
        
        return result