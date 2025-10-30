import sys

def generate_string(N):
    # Find all divisors of N between 1 and 9
    divisors = [j for j in range(1, 10) if N % j == 0]

    # Initialize the result string with '-'
    result = ['-'] * (N + 1)

    # Fill in the result string based on the given conditions
    for i in range(N + 1):
        for j in divisors:
            if i % (N // j) == 0:
                result[i] = str(j)
                break

    # Join the list into a string and print it
    print(''.join(result))

# Read input from stdin
N = int(sys.stdin.read().strip())

# Generate and print the string
generate_string(N)