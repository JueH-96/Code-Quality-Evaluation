# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# B_list will store the sequence B_1, B_2, ..., B_{N-1}
B_list = []

# According to the problem, B_i = A_i * A_{i+1}.
# In Python's 0-indexed list A:
# B_1 corresponds to A[0] * A[1]
# B_2 corresponds to A[1] * A[2]
# ...
# B_{N-1} corresponds to A[N-2] * A[N-1]

# We need to iterate from the first pair (A[0], A[1]) up to the last pair (A[N-2], A[N-1]).
# The loop range for the first element of the pair (A[i]) will be from i=0 to i=N-2.
# range(N - 1) generates integers from 0 up to N-2.
for i in range(N - 1):
    product = A[i] * A[i+1]
    B_list.append(product)

# Print the elements of B_list, separated by spaces.
# The * operator unpacks B_list into individual arguments for the print function.
# print(B_list[0], B_list[1], ..., B_list[len(B_list)-1])
# By default, print separates its arguments with a space.
print(*B_list)