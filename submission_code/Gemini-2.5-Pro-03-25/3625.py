import math

class Solution:
    def canAliceWin(self, n: int) -> bool:
        """
        Determines if Alice can win the stone game given n initial stones.

        Args:
          n: The initial number of stones (1 <= n <= 50).

        Returns:
          True if Alice wins, False otherwise.

        The game rules are:
        1. Alice goes first.
        2. Alice removes 10 stones on her first turn.
        3. Subsequent turns: remove 1 fewer stone than the previous opponent.
        4. The player who cannot make a move loses.

        We can analyze the game turn by turn to find the ranges of 'n' for which Alice wins or loses.

        The sequence of stones to be removed per turn is:
        Turn 1 (Alice): 10
        Turn 2 (Bob):   9 (10 - 1)
        Turn 3 (Alice): 8 (9 - 1)
        Turn 4 (Bob):   7 (8 - 1)
        Turn 5 (Alice): 6 (7 - 1)
        Turn 6 (Bob):   5 (6 - 1)
        Turn 7 (Alice): 4 (5 - 1)
        Turn 8 (Bob):   3 (4 - 1)
        Turn 9 (Alice): 2 (3 - 1)
        Turn 10(Bob):   1 (2 - 1)
        Turn 11(Alice): 0 (1 - 1) -> Cannot remove 0 stones, Alice would lose if it reaches her turn.

        Let's determine the cumulative number of stones removed after each turn:
        End of Turn 1 (Alice removed 10): 10
        End of Turn 2 (Bob removed 9):   10 + 9 = 19
        End of Turn 3 (Alice removed 8): 19 + 8 = 27
        End of Turn 4 (Bob removed 7):   27 + 7 = 34
        End of Turn 5 (Alice removed 6): 34 + 6 = 40
        End of Turn 6 (Bob removed 5):   40 + 5 = 45
        End of Turn 7 (Alice removed 4): 45 + 4 = 49
        End of Turn 8 (Bob removed 3):   49 + 3 = 52 (Exceeds max n=50)
        End of Turn 9 (Alice removed 2): 52 + 2 = 54 (Exceeds max n=50)
        End of Turn 10(Bob removed 1):   54 + 1 = 55 (Exceeds max n=50)

        Now, let's analyze the winning/losing conditions based on n:

        - If n < 10 (i.e., n is in [1, 9]):
          Alice needs to remove 10 stones on Turn 1 but cannot.
          Alice loses. -> Return False.

        - If 10 <= n < 19 (i.e., n is in [10, 18]):
          Alice removes 10 stones (remaining: n - 10).
          Bob needs to remove 9 stones.
          Bob cannot move if remaining < 9, which means n - 10 < 9 => n < 19.
          Bob loses. Alice wins. -> Return True.

        - If 19 <= n < 27 (i.e., n is in [19, 26]):
          Alice removes 10 (remaining: n - 10).
          Bob removes 9 (remaining: n - 19).
          Alice needs to remove 8 stones on Turn 3.
          Alice cannot move if remaining < 8, which means n - 19 < 8 => n < 27.
          Alice loses. -> Return False.

        - If 27 <= n < 34 (i.e., n is in [27, 33]):
          Turns 1, 2, 3 happen (total removed: 27, remaining: n - 27).
          Bob needs to remove 7 stones on Turn 4.
          Bob cannot move if remaining < 7, which means n - 27 < 7 => n < 34.
          Bob loses. Alice wins. -> Return True.

        - If 34 <= n < 40 (i.e., n is in [34, 39]):
          Turns 1, 2, 3, 4 happen (total removed: 34, remaining: n - 34).
          Alice needs to remove 6 stones on Turn 5.
          Alice cannot move if remaining < 6, which means n - 34 < 6 => n < 40.
          Alice loses. -> Return False.

        - If 40 <= n < 45 (i.e., n is in [40, 44]):
          Turns 1-5 happen (total removed: 40, remaining: n - 40).
          Bob needs to remove 5 stones on Turn 6.
          Bob cannot move if remaining < 5, which means n - 40 < 5 => n < 45.
          Bob loses. Alice wins. -> Return True.

        - If 45 <= n < 49 (i.e., n is in [45, 48]):
          Turns 1-6 happen (total removed: 45, remaining: n - 45).
          Alice needs to remove 4 stones on Turn 7.
          Alice cannot move if remaining < 4, which means n - 45 < 4 => n < 49.
          Alice loses. -> Return False.

        - If 49 <= n < 52 (which includes n=49, 50 because constraint n <= 50):
          Turns 1-7 happen (total removed: 49, remaining: n - 49).
          Bob needs to remove 3 stones on Turn 8.
          Bob cannot move if remaining < 3, which means n - 49 < 3 => n < 52.
          Since n <= 50, this applies to n=49, 50.
          Bob loses. Alice wins. -> Return True.

        We can summarize the ranges where Alice loses:
        [1, 9]
        [19, 26]
        [34, 39]
        [45, 48]

        If n falls into any of these ranges, Alice loses. Otherwise (within the constraint 1 <= n <= 50), Alice wins.
        """

        # Check the ranges where Alice loses
        if 1 <= n <= 9:
            return False
        if 19 <= n <= 26:
            return False
        if 34 <= n <= 39:
            return False
        if 45 <= n <= 48:
            return False

        # If n is within the valid range [1, 50] and does not fall into any of Alice's
        # losing ranges identified above, then Alice must win.
        # The winning ranges are implicitly: [10, 18], [27, 33], [40, 44], [49, 50].
        return True