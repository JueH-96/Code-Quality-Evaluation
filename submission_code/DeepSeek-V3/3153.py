class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        # Count the number of set bits at each position
        bit_counts = [0] * 31
        for num in nums:
            for i in range(31):
                if num & (1 << i):
                    bit_counts[i] += 1
        # Reconstruct the numbers with the maximum possible bits
        result = []
        for _ in range(k):
            num = 0
            for i in range(30, -1, -1):
                if bit_counts[i] > 0:
                    num |= (1 << i)
                    bit_counts[i] -= 1
            result.append(num)
        # Calculate the sum of squares
        total = 0
        for num in result:
            total = (total + num * num) % MOD
        return total