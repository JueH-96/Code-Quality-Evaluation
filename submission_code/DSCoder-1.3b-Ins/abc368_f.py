# YOUR CODE HERE

N = int(input().strip())
A = list(map(int, input().strip().split()))

# Check if the number of elements in A is even
if N % 2 == 0:
    print("Anna")
else:
    print("Bruno")