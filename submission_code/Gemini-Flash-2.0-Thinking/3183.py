class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        k_or = 0
        max_bits = 31  # Since nums[i] < 2^31

        for bit in range(max_bits):
            count = 0
            for num in nums:
                if (num >> bit) & 1:
                    count += 1

            if count >= k:
                k_or |= (1 << bit)

        return k_or