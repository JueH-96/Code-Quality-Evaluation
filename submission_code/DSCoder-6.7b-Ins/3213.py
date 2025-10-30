class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_count = [0] * (max(nums) + 1)
        prefix_count = [0] * (n + 1)
        prefix_max = [0] * (n + 1)

        for num in nums:
            max_count[num] += 1

        for i in range(1, len(max_count)):
            max_count[i] = max(max_count[i], max_count[i - 1])

        for i in range(1, n + 1):
            prefix_max[i] = max(prefix_max[i - 1], nums[i - 1])

        for i in range(1, n + 1):
            prefix_count[i] = prefix_count[i - 1] + (nums[i - 1] == prefix_max[i])

        count = 0
        for i in range(k - 1, n):
            if nums[i] == prefix_max[i + 1]:
                count += max_count[nums[i]] - prefix_count[i] + (nums[i] == prefix_max[i - k + 2])
            else:
                count += max_count[nums[i]] - prefix_count[i]

        return count