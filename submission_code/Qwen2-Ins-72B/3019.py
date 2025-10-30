class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        balance = 0
        wildcards = 0
        for move in moves:
            if move == 'L':
                balance -= 1
            elif move == 'R':
                balance += 1
            else:
                wildcards += 1
        
        # The furthest distance is achieved by using all wildcards to maximize the imbalance
        return abs(balance) + wildcards