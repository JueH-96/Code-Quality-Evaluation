import sys

def find_greatest_value(S, N):
    # Helper function to convert binary string to decimal integer
    def binary_to_decimal(binary_str):
        return int(binary_str, 2)

    # Generate all possible binary strings by replacing '?' with '0' or '1'
    def generate_binary_strings(S):
        if '?' not in S:
            return [S]
        return generate_binary_strings(S.replace('?', '0', 1)) + generate_binary_strings(S.replace('?', '1', 1))

    # Generate all possible values
    possible_values = [binary_to_decimal(binary_str) for binary_str in generate_binary_strings(S)]

    # Find the greatest value less than or equal to N
    greatest_value = -1
    for value in possible_values:
        if value <= N:
            greatest_value = max(greatest_value, value)

    return greatest_value

# Read input
S = sys.stdin.readline().strip()
N = int(sys.stdin.readline().strip())

# Find and print the greatest value
result = find_greatest_value(S, N)
print(result)