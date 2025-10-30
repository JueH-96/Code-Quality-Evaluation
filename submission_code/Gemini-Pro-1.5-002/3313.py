class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        neg_count = 0
        for num in nums:
            if num < 0:
                neg_count += 1
            else:
                break
        
        ans = 1
        if neg_count >= k:
            for i in range(k):
                ans *= nums[n - 1 - i]
            return ans
        
        ans = 1
        for i in range(min(neg_count, k)):
            if i % 2 == 0:
                ans *= nums[n - 1 - i]
            else:
                ans *= nums[i]

        if k > neg_count:
            for i in range(k - neg_count):
                ans *= nums[neg_count + i]
        
        if neg_count % 2 == 1 and k > neg_count and 0 in nums:
            temp_ans = 1
            for i in range(min(neg_count + 1, k)):
                if i % 2 == 0:
                    temp_ans *= nums[n - 1 - i]
                else:
                    temp_ans *= nums[i]
            if k > neg_count + 1:
                for i in range(k - (neg_count + 1)):
                    temp_ans *= nums[neg_count + 1 + i]
            ans = max(ans, temp_ans)
            
        
        return ans