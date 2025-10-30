class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        
        # Initialize the maximum beauty to the length of the array
        max_beauty = n
        
        # Iterate through the array
        for i in range(n):
            # Find the range of values that can be assigned to the current element
            min_val = max(0, nums[i] - k)
            max_val = min(10**5, nums[i] + k)
            
            # Find the longest subsequence of equal elements within the range
            max_len = 1
            for j in range(i+1, n):
                if min_val <= nums[j] <= max_val:
                    max_len += 1
                else:
                    break
            
            # Update the maximum beauty
            max_beauty = max(max_beauty, max_len)
        
        return max_beauty