# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

# Create a dictionary to store the position of each person
positions = {person: position for position, person in enumerate(P)}

Q = int(input())

for _ in range(Q):
    A, B = map(int, input().split())
    
    # Get the positions of person A and person B
    pos_A = positions[A]
    pos_B = positions[B]
    
    # The person with the smaller position is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)