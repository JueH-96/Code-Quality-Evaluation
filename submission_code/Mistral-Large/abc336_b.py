import sys

def ctz(N):
    # Convert N to its binary representation and remove the '0b' prefix
    binary_representation = bin(N)[2:]
    # Count the number of consecutive zeros at the end
    count = 0
    for digit in reversed(binary_representation):
        if digit == '0':
            count += 1
        else:
            break
    return count

# Read input from stdin
N = int(sys.stdin.read().strip())
# Print the result
print(ctz(N))