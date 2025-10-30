class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        char_count = {}
        partition_count = 0
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
            # Check if all characters have appeared the same number of times
            if all(value == char_count[char] for value in char_count.values()):
                partition_count += 1
                char_count.clear()  # Reset for the next partition
        return partition_count

# Example usage:
# sol = Solution()
# print(sol.minimumSubstringsInPartition("fabccddg"))  # Output: 3
# print(sol.minimumSubstringsInPartition("abababaccddb"))  # Output: 2