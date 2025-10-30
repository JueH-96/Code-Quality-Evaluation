# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Sort the masses in descending order
A.sort(reverse=True)

# Calculate the minimum possible mass for each person
min_mass = sum(A) // K

# Initialize variables to store the result
result_min_mass = 0
result_cut_lines = 0

# Try to divide the cake into K parts with the minimum possible mass
for i in range(K):
    current_mass = 0
    current_cut_lines = 0
    for j in range(i, N, K):
        current_mass += A[j]
        if j > 0 and A[j] != A[j-1]:
            current_cut_lines += 1
    if current_mass >= min_mass:
        result_min_mass = min_mass
        result_cut_lines = current_cut_lines
        break

# Print the result
print(result_min_mass, result_cut_lines)