class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        # Step 1: Find the dominant element using Boyer-Moore algorithm
        x = None
        count = 0
        for num in nums:
            if count == 0:
                x = num
                count = 1
            else:
                count += 1 if num == x else -1
        
        # Step 2: Calculate the total frequency of the dominant element
        total_freq = nums.count(x)
        n = len(nums)
        
        # Step 3: Track the frequency in the left part and check conditions
        left_count = 0
        for i in range(n - 1):  # i can be at most n-2
            if nums[i] == x:
                left_count += 1
            
            left_length = i + 1
            if left_count * 2 > left_length:
                right_count = total_freq - left_count
                right_length = n - left_length
                if right_count * 2 > right_length:
                    return i
        
        return -1