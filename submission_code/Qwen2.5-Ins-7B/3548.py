class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        if k == 1:
            return 9 * (10 ** (n - 1))
        
        def count_ways(digit_counts, k, n):
            if n == 0:
                return 1
            ways = 0
            for i in range(1, 10):
                if digit_counts[i] > 0:
                    if i % k == 0 or (n % 2 == 1 and i == k):
                        new_counts = digit_counts.copy()
                        new_counts[i] -= 1
                        ways += count_ways(new_counts, k, n - 1)
            return ways
        
        total_ways = 0
        for i in range(1, 10):
            if n % 2 == 1 and i == k:
                total_ways += count_ways({i: 1} | {j: 0 for j in range(10) if j != i}, k, n - 1)
            else:
                total_ways += count_ways({i: 1} | {j: 0 for j in range(10) if j != i}, k, n - 1)
        
        return total_ways