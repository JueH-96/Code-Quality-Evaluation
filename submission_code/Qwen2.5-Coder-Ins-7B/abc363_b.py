# YOUR CODE HERE
import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
T = int(data[1])
P = int(data[2])
L = list(map(int, data[3:]))

# Count the number of people whose hair length is at least T
count = sum(1 for l in L if l >= T)

# If the condition is already satisfied, print 0
if count >= P:
    print(0)
else:
    # Calculate the number of days required
    days = 0
    while count < P:
        count += sum(1 for l in L if l + days >= T)
        days += 1
    print(days)