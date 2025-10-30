# YOUR CODE HERE

N, K = map(int, input().split())
A = list(map(int, input().split()))

# take K cards from the bottom of the stack and place them on top
B = A[-K:] + A[:-K]

# print the integers written on the cards from top to bottom
print(*B)