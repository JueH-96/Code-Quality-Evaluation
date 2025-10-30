class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(count):
            values = set(count.values())
            return len(values) == 1 and 0 in values

        count = [0] * 26
        partitions = 0

        for char in s:
            count[ord(char) - ord('a')] += 1
            if is_balanced(count):
                partitions += 1
                count = [0] * 26

        return partitions