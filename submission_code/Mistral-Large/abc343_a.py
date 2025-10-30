import sys

# Read input from stdin
input_line = sys.stdin.read().strip()
A, B = map(int, input_line.split())

# Calculate A + B
sum_AB = A + B

# Find an integer between 0 and 9 that is not equal to A + B
for i in range(10):
    if i != sum_AB:
        print(i)
        break