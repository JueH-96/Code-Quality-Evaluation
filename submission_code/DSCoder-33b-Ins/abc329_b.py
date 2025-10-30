# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

max_val = max(A)
A.remove(max_val)

print(max(A))