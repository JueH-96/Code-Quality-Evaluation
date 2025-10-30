import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0]); M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))
    # If M == 2, there are only two good sequences:
    #   either 0,1,0,1,... or 1,0,1,0,...
    # You cannot change any bit without breaking adjacency constraint.
    if M == 2:
        # only possible if A == B already
        print(0 if A == B else -1)
        return
    # For M >= 3, one shows that it's always possible
    # and the minimum number of operations is the sum over i
    # of the minimal circular distance between A[i] and B[i].
    total = 0
    for a, b in zip(A, B):
        # clockwise distance: (b - a) mod M
        cw = (b - a) % M
        # counter-clockwise distance: (a - b) mod M
        ccw = (a - b) % M
        total += min(cw, ccw)
    print(total)

if __name__ == "__main__":
    main()