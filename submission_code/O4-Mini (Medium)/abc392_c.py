import sys
import threading

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    # Read P and convert to 0-based indices
    P = [int(x) - 1 for x in data[1:1+n]]
    # Read Q: bib numbers at each position (0-based positions)
    Q = [int(x) for x in data[1+n:1+2*n]]
    # Build inverse map: pos[bib_value - 1] = index (0-based) of that bib
    pos = [0] * n
    for idx, bib in enumerate(Q):
        pos[bib - 1] = idx
    # Compute S for each bib number from 1..n
    # S[b-1] = bib at the person that the person wearing bib b stares at
    S = [0] * n
    for b in range(1, n+1):
        wearer_idx = pos[b-1]
        target_idx = P[wearer_idx]
        S[b-1] = Q[target_idx]
    # Output
    print(" ".join(map(str, S)))

if __name__ == "__main__":
    main()