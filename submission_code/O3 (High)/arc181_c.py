import sys

def main() -> None:
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(int, input_data[1:N+1]))
    Q = list(map(int, input_data[N+1:2*N+1]))
    
    # rp[i] : position (rank) of row i in permutation P (1-based)
    rp = [0]*(N+1)
    for idx, row in enumerate(P, 1):      # idx : 1 .. N
        rp[row] = idx
    
    # cq[j] : position (rank) of column j in permutation Q (1-based)
    cq = [0]*(N+1)
    for idx, col in enumerate(Q, 1):
        cq[col] = idx
    
    thr = N + 1           # constant threshold
  
    out_lines = []
    for i in range(1, N+1):
        row_bits = ['1' if rp[i] + cq[j] >= thr else '0' for j in range(1, N+1)]
        out_lines.append(''.join(row_bits))
    
    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()