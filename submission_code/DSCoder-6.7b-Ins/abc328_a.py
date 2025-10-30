# YOUR CODE HERE

N, X = map(int, input().split())
S = list(map(int, input().split()))

total_score = sum(s for s in S if s <= X)

print(total_score)