class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        for bit in range(30):  # Since nums[i] < 2^30
            mask = 1 << bit
            current_or = 0
            temp = []
            for num in nums:
                current_or |= num
                if (current_or & mask) == 0:
                    temp.append(num)
                else:
                    if len(temp) >= 1:
                        last = temp[-1]
                        combined = last & num
                        temp[-1] = combined
                    else:
                        temp.append(num)
                    current_or = combined
            if len(temp) <= n - k:
                result |= mask
        return result