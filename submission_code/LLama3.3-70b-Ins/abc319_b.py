def find_smallest_divisor(N, i):
    """Find the smallest divisor j of N between 1 and 9 such that i is a multiple of N/j."""
    for j in range(1, 10):
        if N % j == 0 and i % (N // j) == 0:
            return j
    return None

def generate_string(N):
    """Generate the string s_0s_1...s_N."""
    result = []
    for i in range(N + 1):
        divisor = find_smallest_divisor(N, i)
        if divisor is not None:
            result.append(str(divisor))
        else:
            result.append('-')
    return ''.join(result)

# Read input from stdin
N = int(input())

# Generate and print the string
print(generate_string(N))