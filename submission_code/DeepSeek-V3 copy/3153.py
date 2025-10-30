class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Count the number of set bits at each position
        bit_counts = [0] * 31
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_counts[i] += 1
        # Construct the largest possible numbers
        result = []
        for _ in range(k):
            num = 0
            for i in range(31):
                if bit_counts[i] > 0:
                    num |= (1 << i)
                    bit_counts[i] -= 1
            result.append(num)
        # Calculate the sum of squares
        total = sum(x * x for x in result) % MOD
        return total