# YOUR CODE HERE
import sys

# Function to solve a single test case
def solve():
    # Read the integer n for the current test case
    # sys.stdin.readline() is used for potentially faster input reading compared to input()
    n = int(sys.stdin.readline())
    
    # Determine the winner based on the remainder of n when divided by 3.
    # The game logic is as follows:
    # Vanya (First player) starts. Vanya wins if, after his move, the number is divisible by 3.
    # Vova (Second player) plays optimally to prevent Vanya from winning.
    # If 10 moves pass (5 for Vanya, 5 for Vova) and Vanya hasn't won, Vova wins.

    # Case 1: n % 3 == 1
    # Vanya can subtract 1 from n. The new number becomes n-1.
    # Since n % 3 == 1, (n-1) % 3 == 0.
    # Vanya wins on his first move. The First player wins.

    # Case 2: n % 3 == 2
    # Vanya can add 1 to n. The new number becomes n+1.
    # Since n % 3 == 2, (n+1) % 3 == 3 % 3 == 0.
    # Vanya wins on his first move. The First player wins.

    # Case 3: n % 3 == 0
    # Vanya must make a move. He can either add 1 or subtract 1.
    # If Vanya adds 1, the number becomes n+1. (n+1) % 3 == 1.
    # If Vanya subtracts 1, the number becomes n-1. (n-1) % 3 == -1 % 3 == 2.
    # In either case, the number after Vanya's first move is not divisible by 3.
    # Now it's Vova's turn. The current number x is either 1 mod 3 or 2 mod 3.
    #   If x % 3 == 1, Vova can subtract 1. The new number x-1 is divisible by 3.
    #   If x % 3 == 2, Vova can add 1. The new number x+1 is divisible by 3.
    # Vova's optimal strategy is to always make a move that results in a number divisible by 3.
    # By doing so, Vova ensures that Vanya always starts his turn with a number divisible by 3.
    # As established, if Vanya starts with a number divisible by 3, he cannot win on that turn.
    # This pattern repeats: Vanya starts with 0 mod 3, moves to 1 or 2 mod 3. Vova receives 1 or 2 mod 3, moves back to 0 mod 3.
    # Vanya never gets to make a move that results in a number divisible by 3.
    # After 10 total moves, Vanya has not won. Therefore, Vova wins. The Second player wins.

    if n % 3 == 0:
        # If n is divisible by 3, the Second player (Vova) wins
        print("Second")
    else:
        # If n is not divisible by 3 (remainder 1 or 2), the First player (Vanya) wins
        print("First")

# Read the number of test cases
# Reads the first line of input, which contains the integer t
t = int(sys.stdin.readline())

# Iterate through each test case and call the solve function
for _ in range(t):
    solve()