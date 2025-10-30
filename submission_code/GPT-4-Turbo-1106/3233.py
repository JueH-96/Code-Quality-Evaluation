class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            count, distinct_chars = 0, set()
            for char in s:
                distinct_chars.add(char)
                if len(distinct_chars) > k:
                    count += 1
                    distinct_chars = {char}
            return count + 1 if s else 0

        if k == 1:
            # When k is 1, each character must be in its own partition.
            return len(s)
        
        # No need to change any character if the string already has k or fewer distinct characters.
        if len(set(s)) <= k:
            return 1
        
        max_partitions = 0
        for i in range(len(s)):
            for c in range(ord('a'), ord('z') + 1):
                new_char = chr(c)
                if s[i] != new_char:
                    # Change s[i] to new_char and calculate partitions.
                    new_s = s[:i] + new_char + s[i+1:]
                    partitions = count_partitions(new_s, k)
                    max_partitions = max(max_partitions, partitions)
        
        # Check the case without any changes.
        max_partitions = max(max_partitions, count_partitions(s, k))
        
        return max_partitions

# Example usage:
# sol = Solution()
# print(sol.maxPartitionsAfterOperations("accca", 2))  # Output: 3
# print(sol.maxPartitionsAfterOperations("aabaab", 3))  # Output: 1
# print(sol.maxPartitionsAfterOperations("xxyz", 1))  # Output: 4