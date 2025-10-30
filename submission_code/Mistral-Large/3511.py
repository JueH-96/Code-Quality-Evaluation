class Solution:
    def losingPlayer(self, x: int, y: int) -> str:
        # Calculate the maximum number of turns using 75 coins
        turns_with_75 = x

        # Calculate the maximum number of turns using 10 coins
        turns_with_10 = y // 4

        # The total number of turns is the minimum of the two
        total_turns = min(turns_with_75, turns_with_10)

        # If the total number of turns is even, Bob wins; if odd, Alice wins
        if total_turns % 2 == 0:
            return "Bob"
        else:
            return "Alice"