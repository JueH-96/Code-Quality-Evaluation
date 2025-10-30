# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

# Create a mapping from person number to their position (1-indexed)
position = {}
for i in range(N):
    position[P[i]] = i + 1

Q = int(input())
for _ in range(Q):
    A, B = map(int, input().split())
    # Check which person is further to the front (has smaller position)
    if position[A] < position[B]:
        print(A)
    else:
        print(B)