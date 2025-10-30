class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count the moves in each direction
        count_R = moves.count('R')
        count_L = moves.count('L')
        count_underscore = moves.count('_')
        
        # Calculate net displacement from given fixed moves:
        # positive if more 'R's and negative if more 'L's.
        net = count_R - count_L
        
        # To move as far as possible from the origin,
        # use all underscores in the direction that moves you away from the origin.
        # Thus, the maximum distance from the origin is:
        return abs(net) + count_underscore

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.furthestDistanceFromOrigin("L_RL__R"))  # Expected output: 3
    print(sol.furthestDistanceFromOrigin("_R__LL_"))  # Expected output: 5
    print(sol.furthestDistanceFromOrigin("_______"))  # Expected output: 7