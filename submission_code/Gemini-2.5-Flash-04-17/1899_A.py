import sys

# Read the number of test cases
t = int(sys.stdin.readline())

# Process each test case
for _ in range(t):
    # Read the integer n
    n = int(sys.stdin.readline())
    
    # Optimal play analysis:
    # Vanya wins if he can make the number divisible by 3 after his move.
    # Vanya starts.
    # Turn 1 (Vanya): Initial number is n.
    # Vanya can move to n+1 or n-1.
    # If (n + 1) % 3 == 0, Vanya can move to n+1 and win immediately. This happens if n % 3 == 2.
    # If (n - 1) % 3 == 0, Vanya can move to n-1 and win immediately. This happens if n % 3 == 1.
    # If n % 3 == 1 or n % 3 == 2, Vanya can win on his very first move. Since Vanya plays optimally, he will do this.
    # So, if n % 3 != 0 initially, Vanya wins.

    # If n % 3 == 0 initially:
    # Vanya cannot win immediately on his first move, as (n+1)%3 == 1 and (n-1)%3 == 2.
    # Vanya must move to n+1 or n-1. Let the new number be N1. N1 % 3 != 0.
    # Turn 2 (Vova): Vova receives N1 (N1 % 3 != 0). Vova wants to prevent Vanya from winning on his next turn (Turn 3).
    # Vanya wins on Turn 3 if Vanya receives N2 such that N2 % 3 != 0 (allowing Vanya to move to N2 +/- 1 to get a multiple of 3).
    # Vova receives N1 where N1 % 3 != 0. Vova can move to N1+1 or N1-1.
    # If N1 % 3 == 1: Vova can move to N1+1 (mod 3 is 2) or N1-1 (mod 3 is 0). Vova optimally moves to N1-1, making the number N2 % 3 == 0.
    # If N1 % 3 == 2: Vova can move to N1+1 (mod 3 is 0) or N1-1 (mod 3 is 1). Vova optimally moves to N1+1, making the number N2 % 3 == 0.
    # In either case, if Vova receives a number not divisible by 3, he can move to a number divisible by 3.
    # So, if Vanya starts with n % 3 == 0, he moves to N1 % 3 != 0. Vova receives N1 and optimally moves to N2 % 3 == 0. Vanya receives N2 for his Turn 3.
    # Since N2 % 3 == 0, Vanya cannot win immediately on Turn 3. He moves to N3 % 3 != 0. Vova receives N3.
    # This pattern repeats for all of Vanya's turns (moves 0, 2, 4, 6, 8). Vova can always ensure that the number is divisible by 3 at the start of Vanya's turn.
    # Since Vanya cannot win immediately when the number is divisible by 3 at the start of his turn, and Vova can always force this state, Vanya never wins within his 5 turns.
    # The game reaches 10 total moves. Vanya has not won. Vova wins by the time limit.
    # Therefore, if n % 3 == 0 initially, Vova wins.

    if n % 3 == 0:
        print("Second")
    else:
        print("First")