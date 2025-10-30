class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        current_sum = 0
        element_counts = defaultdict(int)
        distinct_elements = set()

        left = 0

        for right in range(len(nums)):
            current_sum += nums[right]
            element_counts[nums[right]] += 1
            distinct_elements.add(nums[right])

            while right - left + 1 > k:
                current_sum -= nums[left]
                element_counts[nums[left]] -= 1
                if element_counts[nums[left]] == 0:
                    distinct_elements.remove(nums[left])
                left += 1

            if right - left + 1 == k and len(distinct_elements) >= m:
                max_sum = max(max_sum, current_sum)

        return max_sum