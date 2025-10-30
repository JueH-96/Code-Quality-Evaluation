class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        left_moves = moves.count('L')
        right_moves = moves.count('R')
        neutral_moves = moves.count('_')
        
        # Calculate the furthest distance considering all neutral moves as right moves
        furthest_right = right_moves + neutral_moves - left_moves
        
        # Calculate the furthest distance considering all neutral moves as left moves
        furthest_left = left_moves + neutral_moves - right_moves
        
        # The furthest distance from the origin is the maximum of these two
        return max(abs(furthest_right), abs(furthest_left))

# Example usage:
# sol = Solution()
# print(sol.furthestDistanceFromOrigin("L_RL__R"))  # Output: 3
# print(sol.furthestDistanceFromOrigin("_R__LL_"))  # Output: 5
# print(sol.furthestDistanceFromOrigin("_______"))  # Output: 7