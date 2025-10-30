from typing import List

class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        # Convert each level into its score:
        # if possible[i] == 1, then the player gains 1 point;
        # if possible[i] == 0, then the player loses 1 point.
        scores = [1 if x == 1 else -1 for x in possible]
        
        # Total score for all levels.
        total = sum(scores)
        
        # Let Alice play the first (i+1) levels (i from 0 to n-2).
        # Then Bob plays the remaining levels.
        # We want: Alice's score > Bob's score.
        # Since Bob's score = total - Alice's score,
        # the condition becomes:
        #   Alice's score > total - Alice's score  →  2 * (Alice's score) > total.
        # Both players must play at least 1 level, so i must be <= n-2.
        prefix = 0
        for i in range(n - 1):
            prefix += scores[i]
            if 2 * prefix > total:
                # Return the number of levels Alice played, which is (i+1)
                return i + 1
        
        # If no valid split is found, return -1.
        return -1

# Example usage and test cases:
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    print(sol.minimumLevels([1,0,1,0]))  # Expected output: 1
    
    # Example 2:
    print(sol.minimumLevels([1,1,1,1,1]))  # Expected output: 3
    
    # Example 3:
    print(sol.minimumLevels([0,0]))  # Expected output: -1