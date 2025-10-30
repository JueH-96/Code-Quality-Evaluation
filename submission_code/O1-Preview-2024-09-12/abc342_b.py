# YOUR CODE HERE
N = int(input())
P_list = list(map(int, input().split()))
person_to_position = {}
for pos, person in enumerate(P_list, start=1):
    person_to_position[person] = pos

Q = int(input())

for _ in range(Q):
    A_i, B_i = map(int, input().split())
    pos_A = person_to_position[A_i]
    pos_B = person_to_position[B_i]
    if pos_A < pos_B:
        print(A_i)
    else:
        print(B_i)