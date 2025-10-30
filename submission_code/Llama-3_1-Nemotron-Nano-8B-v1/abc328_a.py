# YOUR CODE HERE
n, x = map(int, input().split())
s = list(map(int, input().split()))
print(sum(num for num in s if num <= x))