# YOUR CODE HERE
import sys

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
P = int(data[2])
L = list(map(int, data[3:]))

# Count the number of people whose hair length is already at least T
current_count = sum(1 for l in L if l >= T)

# If the condition is already satisfied, print 0
if current_count >= P:
    print(0)
else:
    # Calculate the number of days required for the condition to be satisfied
    days = max(0, T - max(L) - 1)
    while current_count < P:
        days += 1
        current_count = sum(1 for l in L if l + days >= T)
    print(days)