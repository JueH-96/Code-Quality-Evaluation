class Solution:
  def losingPlayer(self, x: int, y: int) -> str:
    """
    Determines the winner of a coin game.
    Alice and Bob take turns. Alice starts.
    In each turn, a player must pick up coins with a total value of 115.
    The only way to make 115 is with one 75-coin and four 10-coins.
    (1 * 75 + 4 * 10 = 75 + 40 = 115).

    Args:
      x: The number of coins with value 75. (Constraints: 1 <= x <= 100)
      y: The number of coins with value 10. (Constraints: 1 <= y <= 100)

    Returns:
      The name of the player who wins the game ("Alice" or "Bob").
    """

    # Each move consumes:
    # - 1 coin of value 75
    # - 4 coins of value 10

    # The number of moves possible based on the supply of 75-coins is `x`.
    # The number of moves possible based on the supply of 10-coins is `y // 4`
    # (since each move requires 4 coins of value 10).

    # The total number of moves that can be made in the game is limited
    # by the resource that runs out first.
    num_total_moves = min(x, y // 4)

    # Alice is player 1, Bob is player 2. They alternate turns.
    # If `num_total_moves` is 0:
    #   Alice (player 1) cannot make the first move. Alice loses. Bob wins.
    # If `num_total_moves` is 1:
    #   Alice makes 1 move. The game effectively ends as resources for another move are gone.
    #   Bob (player 2) is supposed to move next but cannot. Bob loses. Alice wins.
    # If `num_total_moves` is 2:
    #   Alice makes move 1.
    #   Bob makes move 2. The game effectively ends.
    #   Alice (player 1) is supposed to move next but cannot. Alice loses. Bob wins.
    #
    # This pattern continues:
    # - If `num_total_moves` is odd, Alice makes the last possible move.
    #   The next player (Bob) cannot move and thus loses. So, Alice wins.
    # - If `num_total_moves` is even, Bob makes the last possible move (if k>0).
    #   Or, if k=0, Alice cannot make the first move.
    #   In either case, the next player (Alice) cannot move and thus loses. So, Bob wins.
    
    if num_total_moves % 2 == 1:  # Odd number of moves
        return "Alice"
    else:  # Even number of moves (including 0)
        return "Bob"