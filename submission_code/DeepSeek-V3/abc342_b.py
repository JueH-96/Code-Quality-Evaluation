# YOUR CODE HERE
# Read the number of people
N = int(input())
# Read the list of people's positions
P = list(map(int, input().split()))
# Create a dictionary to map person number to their position
person_to_index = {person: idx for idx, person in enumerate(P)}
# Read the number of queries
Q = int(input())
# Process each query
for _ in range(Q):
    A, B = map(int, input().split())
    # Get the positions of A and B
    pos_A = person_to_index[A]
    pos_B = person_to_index[B]
    # Determine which person is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)