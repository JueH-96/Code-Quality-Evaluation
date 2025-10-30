# Read the number of integers
N = int(input())

# Read the integers as a list
A = list(map(int, input().split()))

# Check if all elements are equal
if len(set(A)) == 1:
    print("Yes")
else:
    print("No")