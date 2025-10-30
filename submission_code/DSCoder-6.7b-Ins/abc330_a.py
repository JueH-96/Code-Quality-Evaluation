# YOUR CODE HERE

N, L = map(int, input().split())
scores = list(map(int, input().split()))

passed = [score for score in scores if score >= L]
print(len(passed))