# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    # Determine if Vanya can win in the first move
    if (n + 1) % 3 == 0 or (n - 1) % 3 == 0:
        print("First")
    else:
        # If not, check if the number of moves is odd or even
        # Since Vanya starts, if the number of moves is odd, Vanya wins
        # Otherwise, Vova wins
        # The number of moves is 10, but since Vanya can force a win in the first move if possible
        # Otherwise, it's a matter of parity
        # Since 10 is even, if Vanya cannot win in the first move, Vova will win
        print("Second")