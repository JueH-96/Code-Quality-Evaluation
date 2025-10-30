class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i-1] + nums[i-1]
        
        l = 0
        ans = 0
        for r in range(n):
            mid = (l + r) // 2
            left_count = mid - l
            left_sum = prefix[mid] - prefix[l]
            right_count = r - mid
            right_sum = prefix[r+1] - prefix[mid+1]
            cost = (nums[mid] * left_count - left_sum) + (right_sum - nums[mid] * right_count)
            
            while cost > k:
                l += 1
                if l > r:
                    cost = 0
                    break
                mid = (l + r) // 2
                left_count = mid - l
                left_sum = prefix[mid] - prefix[l]
                right_count = r - mid
                right_sum = prefix[r+1] - prefix[mid+1]
                cost = (nums[mid] * left_count - left_sum) + (right_sum - nums[mid] * right_count)
            
            ans = max(ans, r - l + 1)
        
        return ans