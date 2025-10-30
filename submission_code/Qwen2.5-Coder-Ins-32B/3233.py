class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            count = 0
            distinct_chars = set()
            for char in s:
                if len(distinct_chars) < k:
                    distinct_chars.add(char)
                else:
                    if char not in distinct_chars:
                        count += 1
                        distinct_chars = {char}
            return count + 1 if distinct_chars else count
        
        max_partitions = count_partitions(s, k)
        for i in range(len(s)):
            for new_char in 'abcdefghijklmnopqrstuvwxyz':
                if new_char != s[i]:
                    new_s = s[:i] + new_char + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(new_s, k))
        
        return max_partitions