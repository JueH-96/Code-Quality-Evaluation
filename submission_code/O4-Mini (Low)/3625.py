class Solution:
    def canAliceWin(self, n: int) -> bool:
        # Prefix sums of stones removed on each turn:
        # Turn 1 (Alice): 10
        # Turn 2 (Bob): 10 + 9 = 19
        # Turn 3 (Alice): 10 + 9 + 8 = 27
        # ... down to removing 1 on turn 10 => sum = 55
        thresholds = [10, 19, 27, 34, 40, 45, 49, 52, 54, 55]
        
        # Find the first turn i where n < thresholds[i-1].
        # If that turn is Bob's (i even), Bob can't move and loses => Alice wins.
        # If that turn is Alice's (i odd), Alice can't move and loses.
        for i, t in enumerate(thresholds, start=1):
            if n < t:
                return (i % 2 == 0)
        
        # For n >= 55 (which won't happen here since n <= 50),
        # all moves 10..1 are possible, but under constraints we never reach here.
        return False