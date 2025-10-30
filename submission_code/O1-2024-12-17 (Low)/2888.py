class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # First, find the dominant element in nums. 
        # Because the problem states there is exactly one dominant element, 
        # a Boyer-Moore majority-vote approach suffices.
        candidate = None
        count = 0
        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        
        # candidate is the dominant element
        # Next, calculate the prefix count of how many times candidate has appeared until each index.
        n = len(nums)
        prefix = [0]*n
        prefix[0] = 1 if nums[0] == candidate else 0
        for i in range(1, n):
            prefix[i] = prefix[i-1] + (1 if nums[i] == candidate else 0)
        
        # total number of occurrences of candidate in nums
        total = prefix[-1]  
        
        # We look for the minimum i < n - 1 such that:
        #   prefix[i]*2 > (i + 1) and (total - prefix[i])*2 > (n - i - 1)
        # Because that ensures the subarray [0..i] and the subarray [i+1..n-1] 
        # both have the same dominant element (candidate).
        for i in range(n - 1):
            left_count = prefix[i]
            right_count = total - left_count
            left_len = i + 1
            right_len = n - i - 1
            
            if left_count * 2 > left_len and right_count * 2 > right_len:
                return i
        
        return -1