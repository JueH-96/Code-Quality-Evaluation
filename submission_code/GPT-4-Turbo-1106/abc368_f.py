def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def game_winner(N, A):
    # Count the number of prime numbers in A
    prime_count = sum(1 for a in A if is_prime(a))
    # If the number of prime numbers is even, Anna wins, otherwise Bruno wins
    return "Anna" if prime_count % 2 == 0 else "Bruno"

# Read input from stdin
N = int(input().strip())
A = list(map(int, input().strip().split()))

# Determine the winner and print the result
winner = game_winner(N, A)
print(winner)