# Read N
N = int(input())

# Read the line of people
P = list(map(int, input().split()))

# Create a position map: person_id -> position (1-indexed)
position = {}
for i, person in enumerate(P):
    position[person] = i + 1

# Read Q
Q = int(input())

# Process queries
for _ in range(Q):
    A, B = map(int, input().split())
    
    # Find positions
    pos_A = position[A]
    pos_B = position[B]
    
    # Return the person with smaller position (further to the front)
    if pos_A < pos_B:
        print(A)
    else:
        print(B)