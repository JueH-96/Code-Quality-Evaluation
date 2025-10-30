# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

# Create a mapping from person number to their position in line
position = {}
for i in range(N):
    position[P[i]] = i

Q = int(input())
for _ in range(Q):
    A, B = map(int, input().split())
    
    # Find positions of person A and B
    pos_A = position[A]
    pos_B = position[B]
    
    # The person with smaller position index is further to the front
    if pos_A < pos_B:
        print(A)
    else:
        print(B)