class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        max_num = max(nums)
        n = len(nums)
        left, right = 0, 0
        count = 0
        max_count = 0

        while right < n:
            if nums[right] == max_num:
                count += 1

            while count >= k:
                max_count += n - right
                if nums[left] == max_num:
                    count -= 1
                left += 1

            right += 1

        return max_count