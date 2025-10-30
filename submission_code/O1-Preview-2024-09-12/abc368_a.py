# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))
bottom_K_cards = A[-K:]
remaining_stack = A[:-K]
new_stack = bottom_K_cards + remaining_stack
print(' '.join(map(str, new_stack)))