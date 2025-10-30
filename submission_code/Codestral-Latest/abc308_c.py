import sys
from decimal import Decimal, getcontext

# Set the precision for Decimal to handle large numbers
getcontext().prec = 30

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
people = []

for i in range(N):
    A_i = int(data[2 * i + 1])
    B_i = int(data[2 * i + 2])
    success_rate = Decimal(A_i) / Decimal(A_i + B_i)
    people.append((success_rate, i + 1))

# Sort people by success rate in descending order, and by number in ascending order for ties
people.sort(key=lambda x: (-x[0], x[1]))

# Print the sorted order of people
for _, person_number in people:
    print(person_number, end=' ')