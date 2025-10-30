# YOUR CODE HERE
N, L = map(int, input().split())
A = list(map(int, input().split()))
print(sum(a >= L for a in A))