class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        # Initialize the maximum score to the smallest possible value
        max_score = float('-inf')
        
        # Initialize prefix maximums for the second, third and fourth elements
        prefix_max_2 = [float('-inf')] * len(b)
        prefix_max_3 = [float('-inf')] * len(b)
        prefix_max_4 = [float('-inf')] * len(b)
        
        # Calculate prefix maximums for the second element
        for i in range(1, len(b)):
            prefix_max_2[i] = max(prefix_max_2[i - 1], a[1] * b[i])
        
        # Calculate prefix maximums for the third element
        for i in range(2, len(b)):
            prefix_max_3[i] = max(prefix_max_3[i - 1], prefix_max_2[i - 1] + a[2] * b[i])
        
        # Calculate prefix maximums for the fourth element
        for i in range(3, len(b)):
            prefix_max_4[i] = max(prefix_max_4[i - 1], prefix_max_3[i - 1] + a[3] * b[i])
        
        # Calculate the maximum score by iterating over the first element
        for i_0 in range(len(b) - 3):
            max_score = max(max_score, a[0] * b[i_0] + prefix_max_4[i_0 + 3])
        
        return max_score