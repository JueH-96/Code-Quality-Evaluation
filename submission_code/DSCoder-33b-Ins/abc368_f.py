# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

# If the sum of the list is even, Anna wins. Otherwise, Bruno wins.
if sum(A) % 2 == 0:
    print("Anna")
else:
    print("Bruno")