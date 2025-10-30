class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import Counter

        # Count the frequency of each element in the array
        freq = Counter(nums)

        # Sort the unique elements in descending order
        unique_elements = sorted(freq.keys(), reverse=True)

        max_operations = 0
        target_score = -1

        # Iterate through the unique elements to find the maximum number of operations with the same score
        for i in range(len(unique_elements)):
            for j in range(i, len(unique_elements)):
                score = unique_elements[i] + unique_elements[j]
                if target_score == -1:
                    target_score = score
                if score != target_score:
                    continue

                # Calculate the number of operations possible with the current score
                operations = min(freq[unique_elements[i]], freq[unique_elements[j]])
                if i != j:
                    operations = min(operations, 1)

                max_operations = max(max_operations, operations)

        return max_operations