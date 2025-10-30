# YOUR CODE HERE
def solve():
    n = int(input())
    # The game revolves around the number's remainder when divided by 3.
    # Vanya wins if, after his move, the number is divisible by 3 (i.e., new_n % 3 == 0).
    # Vova wins if 10 moves pass and Vanya hasn't won. Both play optimally.

    # Case 1: n % 3 == 1
    # Vanya can subtract 1. (n-1) % 3 == 0. Vanya wins immediately on his first move.
    
    # Case 2: n % 3 == 2
    # Vanya can add 1. (n+1) % 3 == 0. Vanya wins immediately on his first move.

    # Case 3: n % 3 == 0
    # Vanya must make a move (n+1 or n-1).
    # If Vanya chooses n+1, (n+1) % 3 == 1.
    # If Vanya chooses n-1, (n-1) % 3 == 2.
    # In both scenarios, the number is NOT divisible by 3 after Vanya's move. So Vanya does not win on this turn.
    # Now it's Vova's turn. Vova sees a number X where X % 3 is 1 or 2.
    # Vova plays optimally to prevent Vanya from winning.
    # If X % 3 == 1, Vova subtracts 1 to make it (X-1) % 3 == 0.
    # If X % 3 == 2, Vova adds 1 to make it (X+1) % 3 == 0.
    # Vova can always make the number divisible by 3.
    # When it becomes Vanya's turn again, he will always face a number that is a multiple of 3.
    # This cycle repeats: Vanya always faces a multiple of 3, changes it to a non-multiple of 3,
    # and Vova changes it back to a multiple of 3.
    # Vanya will never be able to make the number divisible by 3 AFTER his move.
    # Thus, if n % 3 == 0 initially, Vanya never wins, and Vova wins after 10 moves.
    
    if n % 3 == 0:
        print("Second")
    else:
        print("First")

# Read the number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    solve()