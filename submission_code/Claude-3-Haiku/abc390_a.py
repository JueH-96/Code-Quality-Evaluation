import sys

# Read input from stdin
input_line = sys.stdin.readline().strip().split()
A = [int(x) for x in input_line]

# Check if the sequence can be sorted in ascending order by performing exactly one swap
can_be_sorted = False
for i in range(len(A) - 1):
    A[i], A[i+1] = A[i+1], A[i]
    if A == sorted(A):
        can_be_sorted = True
        break
    A[i], A[i+1] = A[i+1], A[i]

# Write output to stdout
if can_be_sorted:
    print("Yes")
else:
    print("No")