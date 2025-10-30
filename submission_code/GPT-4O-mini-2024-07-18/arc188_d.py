def count_valid_pairs(N, A, B):
    MOD = 998244353

    # Create a set of used numbers from A and B
    used = set(A)
    for b in B:
        if b != -1:
            used.add(b)

    # Determine the available numbers for b
    available_b = [i for i in range(1, 2 * N + 1) if i not in used]

    # Count how many -1s are in B
    count_neg_ones = B.count(-1)

    # If there are more -1s than available numbers, it's impossible to fill them
    if count_neg_ones > len(available_b):
        return 0

    # Calculate the number of ways to fill the -1s in B
    from math import factorial

    # We can choose any count_neg_ones from available_b and arrange them
    ways_to_fill_b = factorial(len(available_b)) // factorial(len(available_b) - count_neg_ones)

    # The result is the number of ways to fill B
    return ways_to_fill_b % MOD

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:N+1]))
B = list(map(int, data[N+1:2*N+1]))

# Calculate the result
result = count_valid_pairs(N, A, B)

# Print the result
print(result)