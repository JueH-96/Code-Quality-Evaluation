# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

# The sum of all scores must be 0
# So person N's score is the negative of the sum of all other scores
answer = -sum(A)
print(answer)