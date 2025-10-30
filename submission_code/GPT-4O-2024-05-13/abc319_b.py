# YOUR CODE HERE
def generate_string(N):
    result = []
    for i in range(N + 1):
        found = False
        for j in range(1, 10):
            if N % j == 0 and i % (N // j) == 0:
                result.append(str(j))
                found = True
                break
        if not found:
            result.append('-')
    return ''.join(result)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Generate and print the result
print(generate_string(N))