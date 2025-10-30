# YOUR CODE HERE
import sys
import math

def cross(o, a, b):
    return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])

def ccw(a, b, c):
    return cross(a, b, c)

def intersect(a1, a2, b1, b2):
    ccw1 = ccw(a1, a2, b1)
    ccw2 = ccw(a1, a2, b2)
    ccw3 = ccw(b1, b2, a1)
    ccw4 = ccw(b1, b2, a2)
    if ((ccw1 * ccw2) < 0) and ((ccw3 * ccw4) < 0):
        return True
    return False

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = []
    Q = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index+1])
        P.append((A, B))
        index += 2
    for _ in range(N):
        C = int(data[index])
        D = int(data[index+1])
        Q.append((C, D))
        index += 2
    # Sort P and Q based on x-coordinate
    P_sorted = sorted(P, key=lambda x: x[0])
    Q_sorted = sorted(Q, key=lambda x: x[0])
    # Assign Q_sorted to P_sorted in order
    R = []
    for i in range(N):
        R.append(Q_sorted.index(Q[i]) + 1)
    # Check if the assignment is valid
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if intersect(P_sorted[i], Q_sorted[R[i]-1], P_sorted[j], Q_sorted[R[j]-1]):
                valid = False
                break
        if not valid:
            break
    if not valid:
        print(-1)
    else:
        print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()