import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    P = list(map(int, data[1:1+N]))
    Q = list(map(int, data[1+N:1+2*N]))
    # invP[i] = rank of row i in P-order (1-based)
    invP = [0]*(N+1)
    invQ = [0]*(N+1)
    for rnk, x in enumerate(P, start=1):
        invP[x] = rnk
    for rnk, x in enumerate(Q, start=1):
        invQ[x] = rnk
    # Construct A[i][j] = '1' if invP[i] + invQ[j] > N+1 else '0'
    out = []
    for i in range(1, N+1):
        row = []
        ip = invP[i]
        # build row as list of chars
        for j in range(1, N+1):
            if ip + invQ[j] > N+1:
                row.append('1')
            else:
                row.append('0')
        out.append(''.join(row))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()