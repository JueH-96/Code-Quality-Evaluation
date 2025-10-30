import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
P = int(data[2])
F = list(map(int, data[3:]))

# Sort the fares in descending order
F.sort(reverse=True)

# Initialize variables
total_cost = 0
i = 0

# Iterate through the sorted fares
while i < N:
    # Calculate the cost of buying one-day passes for the next D days
    pass_cost = P
    fare_cost = sum(F[i:i+D])

    # If buying passes is cheaper, do it; otherwise, pay the regular fare
    if pass_cost < fare_cost:
        total_cost += pass_cost
        i += D
    else:
        total_cost += F[i]
        i += 1

print(total_cost)