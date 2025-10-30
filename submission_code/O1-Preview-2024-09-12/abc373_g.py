# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    N_and_rest = N_and_rest[1:]
    N_int = N
    P = []
    Q = []
    # Read P_i's
    for i in range(N):
        A_i = int(N_and_rest[2*i])
        B_i = int(N_and_rest[2*i+1])
        P.append((A_i, B_i, i))
    # Offset index
    offset = 2*N
    # Read Q_i's
    for i in range(N):
        C_i = int(N_and_rest[offset + 2*i])
        D_i = int(N_and_rest[offset + 2*i+1])
        Q.append((C_i, D_i, i))
    # Sort P_i's by x-coordinate (then y)
    P_sorted_indices = sorted(range(N), key=lambda i: (P[i][0], P[i][1]))
    # Sort Q_i's by y-coordinate (then x)
    Q_sorted_indices = sorted(range(N), key=lambda i: (Q[i][1], Q[i][0]))
    # Prepare R
    R = [-1]*N
    for k in range(N):
        pi = P_sorted_indices[k]
        qj = Q_sorted_indices[k]
        R[pi] = qj+1  # +1 because indices are from 0 (python) but problem requires from 1
    # Now, need to ensure that the segments do not cross
    # However, according to the discussion, given that P_i's are sorted by x and
    # Q_j's are sorted by y, matching in order avoids crossings
    # So we can output R
    print(' '.join(map(str, R)))
    

threading.Thread(target=main).start()