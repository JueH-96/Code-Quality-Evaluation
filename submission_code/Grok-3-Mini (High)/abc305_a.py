# Read input from stdin
N = int(input())

# Calculate the nearest water station position
lower = (N // 5) * 5
rem = N % 5
if rem <= 2:
    nearest = lower
else:
    nearest = lower + 5

# Output the result to stdout
print(nearest)