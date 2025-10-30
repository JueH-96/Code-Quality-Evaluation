import sys
from itertools import permutations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])

dishes = []
for i in range(N):
    A_i = int(data[3 + 2 * i])
    B_i = int(data[4 + 2 * i])
    dishes.append((A_i, B_i))

# Function to calculate the maximum number of dishes Snuke can eat
def max_dishes(dishes, X, Y):
    max_count = 0
    for order in permutations(dishes):
        sweetness = 0
        saltiness = 0
        count = 0
        for dish in order:
            sweetness += dish[0]
            saltiness += dish[1]
            if sweetness > X or saltiness > Y:
                break
            count += 1
        max_count = max(max_count, count)
    return max_count

# Calculate and print the result
result = max_dishes(dishes, X, Y)
print(result)