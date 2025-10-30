import sys

# Global constants
MAX_VAL = 100000

# Global arrays for precomputation
# spf[i] will store the smallest prime factor of i
spf = [0] * (MAX_VAL + 1)
# omega_counts[i] will store Ω(i), the number of prime factors of i counted with multiplicity
omega_counts = [0] * (MAX_VAL + 1)

def precompute_all():
    """
    Precomputes the Smallest Prime Factor (SPF) for numbers up to MAX_VAL
    and the count of prime factors (Omega function, Ω(n)) for these numbers.
    This function is called once when the script is loaded.
    """
    
    # Initialize spf array: spf[i] = i means i is prime (or not yet processed by sieve)
    for i in range(MAX_VAL + 1):
        spf[i] = i
    
    # Sieve to calculate SPF
    # Iterate from 2 up to sqrt(MAX_VAL)
    # For each prime i, mark its multiples j by setting spf[j] = i
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:  # Update spf[j] only if it hasn't been set by a smaller prime
                    spf[j] = i
    
    # Calculate omega_counts (number of prime factors counted with multiplicity, Ω(n))
    # Base case: omega_counts[1] = 0 (Ω(1) = 0, as 1 has no prime factors).
    # omega_counts[0] is unused and remains 0.
    # omega_counts[1] is already 0 due to the list initialization: `[0] * (MAX_VAL + 1)`.
    
    # For i from 2 to MAX_VAL: Ω(i) = Ω(i / spf[i]) + 1
    # This recursive relation works because spf[i] is one prime factor of i.
    for i in range(2, MAX_VAL + 1):
        omega_counts[i] = 1 + omega_counts[i // spf[i]]

# Perform precomputation once when the script is loaded/imported
precompute_all()

def solve_problem():
    """
    Reads input for a single test case, calculates the nim-sum using 
    precomputed omega_counts, and prints the winner.
    """
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    nim_sum = 0
    for value_a in A:
        nim_sum ^= omega_counts[value_a]

    if nim_sum > 0:
        sys.stdout.write("Anna
")
    else:
        sys.stdout.write("Bruno
")

# Main execution block
if __name__ == '__main__':
    solve_problem()