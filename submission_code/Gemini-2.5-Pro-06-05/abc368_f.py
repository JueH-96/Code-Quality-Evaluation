import sys

def main():
    """
    Solves the Divisor Game problem using the Sprague-Grundy theorem.
    """
    # The maximum possible value for an element in A is 10^5.
    MAX_A = 10**5

    # Step 1: Precompute the smallest prime factor (spf) for all numbers up to MAX_A.
    # This is done using a sieve, which is an efficient method.
    # spf[i] will store the smallest prime factor of i.
    spf = list(range(MAX_A + 1))
    for i in range(2, int(MAX_A**0.5) + 1):
        if spf[i] == i:  # This condition checks if i is a prime number.
            # Mark i as the smallest prime factor for all its multiples.
            for j in range(i * i, MAX_A + 1, i):
                # Update only if j's smallest prime factor has not been found yet.
                if spf[j] == j:
                    spf[j] = i

    # Step 2: Precompute Omega(n), the number of prime factors of n (with multiplicity).
    # This value serves as the Grundy number (g-number) for a subgame on integer n.
    # The recurrence Omega(n) = Omega(n / spf[n]) + 1 is used for efficient calculation.
    omega = [0] * (MAX_A + 1)
    for i in range(2, MAX_A + 1):
        omega[i] = omega[i // spf[i]] + 1

    # Step 3: Read problem inputs from standard input.
    # N is read according to the problem format, but its value is also given by len(A).
    try:
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle cases with empty or malformed input gracefully.
        return

    # Step 4: Calculate the nim-sum of the game.
    # The nim-sum is the XOR sum of the Grundy numbers of the individual subgames.
    nim_sum = 0
    for a in A:
        nim_sum ^= omega[a]

    # Step 5: Determine the winner.
    # According to the Sprague-Grundy theorem, the first player (Anna) has a winning
    # strategy if the initial nim-sum is non-zero. Otherwise, the second player (Bruno) wins.
    if nim_sum == 0:
        print("Bruno")
    else:
        print("Anna")

if __name__ == "__main__":
    main()