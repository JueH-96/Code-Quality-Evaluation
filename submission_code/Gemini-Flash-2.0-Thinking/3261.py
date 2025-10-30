class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)

        def can_achieve(target_or):
            num_subarrays = 0
            current_subarray_and = -1

            for num in nums:
                if current_subarray_and == -1:
                    current_subarray_and = num
                    if current_subarray_and & target_or != current_subarray_and:
                        return False
                else:
                    if (current_subarray_and & num) & target_or == (current_subarray_and & num):
                        current_subarray_and &= num
                    else:
                        num_subarrays += 1
                        current_subarray_and = num
                        if current_subarray_and & target_or != current_subarray_and:
                            return False

            if current_subarray_and != -1:
                num_subarrays += 1

            return num_subarrays <= n - k

        overall_or = 0
        for num in nums:
            overall_or |= num

        left, right = 0, overall_or
        ans = overall_or

        while left <= right:
            mid = (left + right) // 2
            if can_achieve(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans