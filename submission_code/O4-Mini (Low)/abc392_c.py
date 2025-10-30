import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    # P_list: P[1..n], Q_list: Q[1..n]
    # data layout: [n, P1, P2, ..., Pn, Q1, Q2, ..., Qn]
    P_list = list(map(int, data[1:1+n]))
    Q_list = list(map(int, data[1+n:1+2*n]))
    
    # Build inverse of Q: invQ[bib_number] = position of that bib
    invQ = [0] * (n + 1)
    for pos, bib in enumerate(Q_list, start=1):
        invQ[bib] = pos
    
    # For each bib number i, find the person staring at:
    # owner_pos = invQ[i]
    # target_pos = P_list[owner_pos-1]
    # S_i = Q_list[target_pos-1]
    result = [0] * n
    for bib in range(1, n+1):
        owner_pos = invQ[bib]
        target_pos = P_list[owner_pos - 1]
        result[bib - 1] = Q_list[target_pos - 1]
    
    # Output
    sys.stdout.write(" ".join(map(str, result)))

if __name__ == "__main__":
    main()