class Solution:
    def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
        n = len(word)
        substring_count = {}

        # Step 1: Break the string into substrings of length k
        for i in range(0, n, k):
            substring = word[i:i+k]
            if substring in substring_count:
                substring_count[substring] += 1
            else:
                substring_count[substring] = 1

        # Step 2: Find the substring with the maximum count
        max_count = 0
        for count in substring_count.values():
            if count > max_count:
                max_count = count

        # Step 3: Calculate the number of operations needed
        total_substrings = n // k
        operations_needed = total_substrings - max_count

        return operations_needed