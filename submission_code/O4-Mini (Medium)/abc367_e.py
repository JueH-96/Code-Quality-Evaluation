import sys
from array import array

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    # Read X, convert to 0-based, store in an unsigned‐int array
    X0 = array('I', (int(next(it)) - 1 for _ in range(N)))
    # Read A
    A = [int(next(it)) for _ in range(N)]
    # If K == 0, no moves
    if K == 0:
        print(*A)
        return
    # Number of bits needed
    maxlg = K.bit_length()
    # Build binary‐lifting table: up[j][i] = X applied 2^j times to i
    up = [X0]
    for j in range(1, maxlg):
        prev = up[j-1]
        # Build the next layer as an unsigned‐int array
        curr = array('I', (prev[prev[i]] for i in range(N)))
        up.append(curr)
    # cur[i] will track the position we end up at starting from i
    cur = array('I', range(N))
    # Apply jumps for each set bit of K
    for j in range(maxlg):
        if (K >> j) & 1:
            pj = up[j]
            for i in range(N):
                cur[i] = pj[cur[i]]
    # Build and print the resulting sequence
    out = [''] * N
    for i in range(N):
        out[i] = str(A[cur[i]])
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()