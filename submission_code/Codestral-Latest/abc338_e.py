# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
chords = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]

def find_intersection(chords):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    for i in range(N):
        for j in range(i+1, N):
            A, B = chords[i]
            C, D = chords[j]
            if intersect((A, 0), (B, 0), (C, 0), (D, 0)):
                return True
    return False

if find_intersection(chords):
    print("Yes")
else:
    print("No")