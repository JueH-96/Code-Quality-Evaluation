# YOUR CODE HERE
# Read N
N = int(input())

# Read the positions
P = list(map(int, input().split()))

# Create a mapping from person number to position (1-indexed)
position = {}
for i in range(N):
    position[P[i]] = i + 1

# Read Q
Q = int(input())

# Process queries
for _ in range(Q):
    A, B = map(int, input().split())
    
    # Find positions of person A and person B
    pos_A = position[A]
    pos_B = position[B]
    
    # The person with smaller position is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)