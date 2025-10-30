def count_possible_xor_values(N, A):
    from functools import reduce
    from itertools import combinations

    # Calculate the total sum of stones
    total_sum = sum(A)

    # To store all possible XOR values
    possible_xor_values = set()

    # We can generate all possible subsets of the bags
    for i in range(N + 1):
        for combo in combinations(A, i):
            # Calculate the XOR of the current combination
            current_xor = reduce(lambda x, y: x ^ y, combo, 0)
            # The remaining stones will be total_sum - sum(combo)
            remaining_xor = total_sum - sum(combo)
            # The final XOR value is current_xor ^ remaining_xor
            final_xor = current_xor ^ remaining_xor
            possible_xor_values.add(final_xor)

    return len(possible_xor_values)

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Get the result
result = count_possible_xor_values(N, A)

# Print the result
print(result)