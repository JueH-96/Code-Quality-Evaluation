class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        # Count forced lefts, rights, and flexible moves
        lefts = moves.count('L')
        rights = moves.count('R')
        flex = moves.count('_')
        
        # If we direct all flexible moves to the right:
        #   net displacement = (rights + flex) - lefts
        net_if_right = rights + flex - lefts
        # If we direct all flexible moves to the left:
        #   net displacement = (lefts + flex) - rights
        net_if_left = lefts + flex - rights
        
        # Return the maximum absolute net displacement
        return max(net_if_right, net_if_left)