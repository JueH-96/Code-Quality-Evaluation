N = int(input())
P = list(map(int, input().split()))
Q = int(input())
AB = [list(map(int, input().split())) for _ in range(Q)]

# Create a dictionary to map person number to position
position = {person: i for i, person in enumerate(P)}

for A, B in AB:
    if position[A] < position[B]:
        print(A)
    else:
        print(B)