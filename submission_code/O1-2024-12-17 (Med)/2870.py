class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = -1

        # We only need to start checking from each position if there's
        # a +1 difference to the next element (since s_1 = s_0 + 1).
        for i in range(n - 1):
            if nums[i + 1] == nums[i] + 1:  # valid start of an alternating subarray
                length = 2
                prev = nums[i + 1]
                expected_diff = -1  # next difference should be -1

                # Continue extending the subarray
                for j in range(i + 2, n):
                    if expected_diff == -1:
                        if nums[j] == prev - 1:
                            length += 1
                            prev = nums[j]
                            expected_diff = 1  # switch expected diff
                        else:
                            break
                    else:  # expected_diff == 1
                        if nums[j] == prev + 1:
                            length += 1
                            prev = nums[j]
                            expected_diff = -1
                        else:
                            break

                max_len = max(max_len, length)

        return max_len