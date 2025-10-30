# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
P = int(data[2])
L = list(map(int, data[3:]))

# Count the number of people with hair length >= T initially
count = sum(1 for l in L if l >= T)

# If the condition is already satisfied, print 0
if count >= P:
    print(0)
else:
    # Simulate the growth of hair until the condition is satisfied
    days = 0
    while count < P:
        days += 1
        L = [l + 1 for l in L]
        count = sum(1 for l in L if l >= T)
    print(days)