class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the number of 'L', 'R', and '_' in the string
        left = moves.count('L')
        right = moves.count('R')
        blanks = moves.count('_')
        
        # Option 1: Convert all blanks to right moves.
        # This makes the net movement: right + blanks (all right) - left.
        # Option 2: Convert all blanks to left moves.
        # This makes the net movement: left + blanks (all left) - right.
        # The answer is the maximum absolute displacement from the origin.
        return max(right + blanks - left, left + blanks - right)
        
# Optional testing:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    moves = "L_RL__R"
    print(sol.furthestDistanceFromOrigin(moves))  # Expected output: 3
    
    # Example 2
    moves = "_R__LL_"
    print(sol.furthestDistanceFromOrigin(moves))  # Expected output: 5
    
    # Example 3
    moves = "_______"
    print(sol.furthestDistanceFromOrigin(moves))  # Expected output: 7