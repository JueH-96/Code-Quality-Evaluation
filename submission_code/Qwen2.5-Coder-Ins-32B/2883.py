class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        def is_power_of_five(num):
            while num > 1:
                if num % 5 != 0:
                    return False
                num //= 5
            return num == 1

        def dfs(start):
            if start == len(s):
                return 0
            if s[start] == '0':
                return float('inf')
            
            min_partitions = float('inf')
            current_number = 0
            for end in range(start, len(s)):
                current_number = (current_number << 1) | (int(s[end]))
                if is_power_of_five(current_number):
                    min_partitions = min(min_partitions, 1 + dfs(end + 1))
            return min_partitions

        result = dfs(0)
        return result if result != float('inf') else -1