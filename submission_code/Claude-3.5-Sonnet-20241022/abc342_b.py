N = int(input())
P = list(map(int, input().split()))
Q = int(input())

# Create position lookup array
pos = [0] * (N+1)
for i in range(N):
    pos[P[i]] = i

for _ in range(Q):
    A, B = map(int, input().split())
    # Compare positions and print person number of one standing further front
    if pos[A] < pos[B]:
        print(A)
    else:
        print(B)