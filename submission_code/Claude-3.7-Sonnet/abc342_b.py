# Read the number of people
N = int(input())

# Read the person at each position
P = list(map(int, input().split()))

# Create a mapping from person to position
person_to_position = {}
for position, person in enumerate(P, 1):
    person_to_position[person] = position

# Read the number of queries
Q = int(input())

# Process the queries
for _ in range(Q):
    A, B = map(int, input().split())
    if person_to_position[A] < person_to_position[B]:
        print(A)
    else:
        print(B)