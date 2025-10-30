# YOUR CODE HERE
n, k = map(int, input().split())
a = set(map(int, input().split()))
print(sum(i for i in range(1, k+1) if i not in a))