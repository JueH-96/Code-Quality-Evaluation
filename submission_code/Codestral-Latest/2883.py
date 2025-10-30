class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Helper function to check if a string is a power of 5
        def is_power_of_5(binary_str):
            if binary_str.startswith('0'):
                return False
            num = int(binary_str, 2)
            while num > 1:
                if num % 5 != 0:
                    return False
                num //= 5
            return num == 1

        # Memoization dictionary to store results of subproblems
        memo = {}

        def dp(start):
            if start == len(s):
                return 0
            if start in memo:
                return memo[start]

            min_partitions = float('inf')
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if is_power_of_5(substring):
                    partitions = dp(end)
                    if partitions != -1:
                        min_partitions = min(min_partitions, partitions + 1)

            memo[start] = min_partitions if min_partitions != float('inf') else -1
            return memo[start]

        result = dp(0)
        return result if result != float('inf') else -1