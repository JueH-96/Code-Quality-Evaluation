class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        count = collections.Counter(nums[:k])
        cnt, i = sum(count.values()), k
        for j in range(k, len(nums)):
            if count[nums[j-k]] == 1:
                cnt -= 1
            count[nums[j-k]] -= 1
            count[nums[j]] += 1
            if count[nums[j]] == 1:
                cnt += 1
            if cnt == k:
                return j - k + 1
        return j - k + 1