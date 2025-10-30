class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        
        # Create a frequency array to keep track of the count of each element
        freq = [0] * (max(nums) + 1)
        
        for num in nums:
            freq[num] += 1
        
        for l, r in queries:
            # Decrement the count of each element in the range [l, r]
            for i in range(l, r+1):
                if nums[i] > 0:
                    freq[nums[i]] -= 1
                    nums[i] -= 1
        
        # Check if all elements in the frequency array are 0
        return all(count == 0 for count in freq)