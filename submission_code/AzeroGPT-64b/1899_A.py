class GameOutcome:
    def __init__(self, test_cases):
        self.test_cases = test_cases

    def determine_winner(self):
        """
        Determines the winner of the game based on the given integer n.
        """
        results = []
        for n in self.test_cases:
            # If n is of the form 6k + 5, Vanya wins on the first move.
            # Otherwise, if n is of the form 6k + 4, Vova can force a win.
            if n % 6 == 5:
                results.append("First")
            # If we reach move 10 and haven't won, Vova wins by default.
            elif n + 9 < 3 or (n - 1 + 9) % 3 == 0:
                results.append("Second")
            else:
                # If none of the above conditions are met, check if Vanya can win by reaching a number of the form 6k + 5.
                can_reach_winning_position = False
                for move in range(1, 10):  # Check for all possible moves up to the 9th move.
                    if (n + move) % 6 == 5 or (n - 1 + move) % 6 == 5:
                        can_reach_winning_position = True
                        break
                if can_reach_winning_position:
                    results.append("First")
                else:
                    results.append("Second")
        return results


# Reading from stdin and writing to stdout for judging purposes
if __name__ == "__main__":
    import sys

    t = int(sys.stdin.readline().strip())
    test_cases = [int(sys.stdin.readline().strip()) for _ in range(t)]
    game = GameOutcome(test_cases)
    results = game.determine_winner()
    for result in results:
        sys.stdout.write(result + "
")