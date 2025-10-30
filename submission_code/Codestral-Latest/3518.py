class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Sort the indices of b in descending order based on the value of b
        indices = sorted(range(len(b)), key=lambda i: b[i], reverse=True)

        # Initialize the maximum score
        max_score = float('-inf')

        # Iterate through the sorted indices and calculate the score for each combination
        for i in range(len(indices) - 3):
            for j in range(i + 1, len(indices) - 2):
                for k in range(j + 1, len(indices) - 1):
                    for l in range(k + 1, len(indices)):
                        score = a[0] * b[indices[i]] + a[1] * b[indices[j]] + a[2] * b[indices[k]] + a[3] * b[indices[l]]
                        max_score = max(max_score, score)

        return max_score