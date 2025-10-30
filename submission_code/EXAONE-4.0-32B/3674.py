class Solution:
    def countNonDecreasingSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        left = 0
        for right in range(n):
            if right == left:
                cost = 0
                last_val = nums[left]
            else:
                if nums[right] >= last_val:
                    last_val = nums[right]
                else:
                    cost += last_val - nums[right]
            while cost > k and left < right:
                left += 1
                cost = 0
                last_val = nums[left]
                for i in range(left + 1, right + 1):
                    if nums[i] >= last_val:
                        last_val = nums[i]
                    else:
                        cost += last_val - nums[i]
            if cost <= k:
                ans += right - left + 1
        return ans