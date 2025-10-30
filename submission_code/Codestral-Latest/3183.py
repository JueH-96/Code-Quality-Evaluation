class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        # Initialize an array to count the number of times each bit is set
        bit_count = [0] * 32

        # Count the number of times each bit is set across all numbers in nums
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1

        # Initialize the result
        result = 0

        # Construct the result by setting bits that are set in at least k elements
        for i in range(32):
            if bit_count[i] >= k:
                result |= (1 << i)

        return result