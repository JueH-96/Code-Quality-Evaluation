# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

min_val = min(A)
max_val = max(A)

if max_val - min_val > 1:
    print(len(A) * (max_val - min_val - 1))
else:
    print(0)