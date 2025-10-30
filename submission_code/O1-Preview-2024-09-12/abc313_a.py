# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))
max_other = max(P[1:])
x = max(0, max_other - P[0] + 1)
print(x)