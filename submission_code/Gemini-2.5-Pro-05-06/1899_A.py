# Read the number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    # Read the integer n for the current test case
    n = int(input())
    
    # Determine the winner based on n modulo 3
    if n % 3 == 0:
        # If n is divisible by 3, Vanya cannot win on his first move.
        # Vanya must make it n+1 or n-1, neither of which is divisible by 3.
        # Vova can then always restore the state to a number divisible by 3.
        # This cycle repeats: Vanya gets a number divisible by 3, can't win.
        # Vova gets a number not divisible by 3, makes it divisible by 3.
        # Vanya never wins. After 10 moves, Vova wins.
        print("Second")
    else:
        # If n % 3 is 1 or 2, Vanya can win on his first move.
        # If n % 3 == 1, Vanya subtracts 1, making it (n-1) % 3 == 0.
        # If n % 3 == 2, Vanya adds 1, making it (n+1) % 3 == 0.
        # Vanya wins.
        print("First")