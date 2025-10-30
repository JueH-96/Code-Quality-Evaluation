import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Calculate the sum of all elements in A
total_sum = sum(A)

# For each element in A, calculate the sum of elements greater than it
B = []
for i in range(N):
    sum_greater = total_sum - A[i] - sum(a for a in A if a <= A[i])
    B.append(sum_greater)

# Print the results
print(" ".join(map(str, B)))