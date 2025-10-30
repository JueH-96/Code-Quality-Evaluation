# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1000000)
    N = int(sys.stdin.readline())
    P = []
    for i in range(N):
        A, B = map(int, sys.stdin.readline().split())
        P.append((A, B, i))
    Q = []
    for i in range(N):
        C, D = map(int, sys.stdin.readline().split())
        Q.append((C, D, i+1))  # 1-based indexing for R

    # Sort P by x ascending, then y ascending
    P_sorted = sorted(P, key=lambda x: (x[0], x[1]))
    # Sort Q by x ascending, then y ascending
    Q_sorted = sorted(Q, key=lambda x: (x[0], x[1]))

    # Assign R_sorted_p[i] = Q_sorted[i].index
    R_sorted = [q[2] for q in Q_sorted]

    # Now, map back to original P indices
    R = [0]*N
    for i in range(N):
        original_p_index = P_sorted[i][2]
        R[original_p_index] = R_sorted[i]

    # To ensure no crossing, we might need to verify, but based on sorting, it should be non-crossing

    # Output R
    print(' '.join(map(str, R)))

if __name__ == "__main__":
    main()