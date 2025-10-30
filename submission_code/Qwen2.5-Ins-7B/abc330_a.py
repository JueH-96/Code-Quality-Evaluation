# YOUR CODE HERE
n, l = map(int, input().split())
scores = list(map(int, input().split()))
passed = sum(1 for score in scores if score >= l)
print(passed)