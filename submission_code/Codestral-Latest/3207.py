class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        # Find the minimum length among the three strings
        min_length = min(len(s1), len(s2), len(s3))

        # Check the common prefix length
        for i in range(min_length):
            if s1[i] != s2[i] or s1[i] != s3[i]:
                common_length = i
                break
        else:
            common_length = min_length

        # If there is no common prefix, return -1
        if common_length == 0:
            return -1

        # Calculate the total number of operations needed
        total_operations = (len(s1) - common_length) + (len(s2) - common_length) + (len(s3) - common_length)

        return total_operations