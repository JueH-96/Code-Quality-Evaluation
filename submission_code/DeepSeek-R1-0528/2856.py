class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        total_distinct = len(set(nums))
        n = len(nums)
        ans = 0
        for left in range(n):
            freq = {}
            cur_distinct = 0
            for right in range(left, n):
                num = nums[right]
                freq[num] = freq.get(num, 0) + 1
                if freq[num] == 1:
                    cur_distinct += 1
                if cur_distinct == total_distinct:
                    ans += (n - right)
                    break
        return ans