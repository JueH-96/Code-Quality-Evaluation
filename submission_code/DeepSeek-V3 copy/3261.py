class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        result = 0
        for bit in range(30):  # Since nums[i] < 2^30
            mask = 1 << bit
            current_or = 0
            temp_k = k
            for num in nums:
                current_or |= num
                if (current_or & mask) == 0:
                    continue
                if temp_k > 0:
                    current_or &= num
                    temp_k -= 1
            if (current_or & mask) != 0:
                result |= mask
        return result