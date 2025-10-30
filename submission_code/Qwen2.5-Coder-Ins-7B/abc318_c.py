# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
P = int(data[2])
F = list(map(int, data[3:]))

# Sort the fares in descending order
F.sort(reverse=True)

# Calculate the minimum cost
cost = 0
i = 0
while i < N:
    if i + D <= N:
        # Buy a batch of one-day passes
        cost += P
        i += D
    else:
        # Pay the regular fare for the remaining days
        cost += sum(F[i:])
        break

print(cost)