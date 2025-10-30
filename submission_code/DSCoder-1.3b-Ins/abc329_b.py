# YOUR CODE HERE

N = int(input().strip())
A = list(map(int, input().strip().split()))

max_val = max(A)
A.remove(max_val)

print(max(A))