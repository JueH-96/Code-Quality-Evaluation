class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        def count_partitions(s, k):
            count = 0
            curr_distinct = 0
            char_count = [0] * 26
            left = 0

            for right in range(len(s)):
                if char_count[ord(s[right]) - ord('a')] == 0:
                    curr_distinct += 1
                char_count[ord(s[right]) - ord('a')] += 1

                while curr_distinct > k:
                    char_count[ord(s[left]) - ord('a')] -= 1
                    if char_count[ord(s[left]) - ord('a')] == 0:
                        curr_distinct -= 1
                    left += 1

                if right == len(s) - 1 or curr_distinct == k:
                    count += 1
                    left = right + 1
                    curr_distinct = 0
                    char_count = [0] * 26

            return count

        max_partitions = count_partitions(s, k)

        for i in range(len(s)):
            for c in range(ord('a'), ord('z') + 1):
                if chr(c) != s[i]:
                    new_s = s[:i] + chr(c) + s[i+1:]
                    max_partitions = max(max_partitions, count_partitions(new_s, k))

        return max_partitions