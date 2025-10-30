# YOUR CODE HERE

N, L = map(int, input().split())
scores = list(map(int, input().split()))

passing_students = [score >= L for score in scores]
print(sum(passing_students))