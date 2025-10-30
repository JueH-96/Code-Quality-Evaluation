# YOUR CODE HERE
import sys

input = sys.stdin.read().split()
N = int(input[0])
T = int(input[1])
P = int(input[2])
L = list(map(int, input[3:]))

days = 0
while True:
    count = sum(1 for length in L if length + days >= T)
    if count >= P:
        break
    days += 1

print(days)