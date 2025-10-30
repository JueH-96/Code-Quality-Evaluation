# YOUR CODE HERE
def count_expressible_numbers(N):
    expressible_numbers = set()
    a = 2
    while a * a <= N:
        power = a * a
        while power <= N:
            expressible_numbers.add(power)
            power *= a
        a += 1
    return len(expressible_numbers)

# Read input
import sys
input = sys.stdin.read
N = int(input().strip())

# Calculate and print the result
result = count_expressible_numbers(N)
print(result)