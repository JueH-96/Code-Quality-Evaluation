import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
cups = []

for i in range(1, N * 2 + 1, 2):
    F_i = int(data[i])
    S_i = int(data[i + 1])
    cups.append((F_i, S_i))

# Dictionary to store the maximum deliciousness for each flavor
flavor_dict = defaultdict(int)

# Maximum deliciousness of any cup
max_deliciousness = 0

# Maximum satisfaction
max_satisfaction = 0

for F_i, S_i in cups:
    if S_i > flavor_dict[F_i]:
        flavor_dict[F_i] = S_i
    if S_i > max_deliciousness:
        max_deliciousness = S_i

# Calculate the maximum satisfaction
for F_i, S_i in cups:
    if S_i == max_deliciousness:
        # If the current cup has the maximum deliciousness, check for different flavors
        for F_j, S_j in flavor_dict.items():
            if F_j != F_i:
                max_satisfaction = max(max_satisfaction, S_i + S_j)
    else:
        # If the current cup does not have the maximum deliciousness, check for the same flavor
        if S_i + flavor_dict[F_i] / 2 > max_satisfaction:
            max_satisfaction = S_i + flavor_dict[F_i] / 2

print(int(max_satisfaction))