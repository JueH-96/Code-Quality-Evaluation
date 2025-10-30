# YOUR CODE HERE
N, L = map(int, input().split())
A = list(map(int, input().split()))
count = sum(1 for a in A if a >= L)
print(count)