class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        n = len(b)
        
        # Sort indices of b based on their values
        indices = sorted(range(n), key=lambda x: b[x])
        
        # Initialize variables to keep track of the best indices
        best = [0] * 4
        current_max = float('-inf')
        
        # Iterate through all possible combinations of 4 indices
        for i in range(n-3):
            for j in range(i+1, n-2):
                for k in range(j+1, n-1):
                    for l in range(k+1, n):
                        # Calculate the score for the current combination
                        score = a[0] * b[indices[i]] + a[1] * b[indices[j]] + a[2] * b[indices[k]] + a[3] * b[indices[l]]
                        
                        # Update the best score if the current score is higher
                        if score > current_max:
                            current_max = score
                            best = [indices[i], indices[j], indices[k], indices[l]]
        
        # Calculate and return the final maximum score
        return a[0] * b[best[0]] + a[1] * b[best[1]] + a[2] * b[best[2]] + a[3] * b[best[3]]