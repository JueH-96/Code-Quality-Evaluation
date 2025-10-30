# YOUR CODE HERE
import sys

def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    S = sys.stdin.readline().strip()
    L = len(S)
    Q_line = sys.stdin.readline()
    while Q_line.strip() == '':
        Q_line = sys.stdin.readline()
    Q = int(Q_line)
    K_line = []
    while len(K_line) < Q:
        line = sys.stdin.readline()
        if not line:
            break
        parts = line.strip().split()
        if parts:
            K_line.extend(parts)
    K_list = list(map(int, K_line[:Q]))
    
    toggled_S = [c.swapcase() for c in S]
    result = []
    for K in K_list:
        k = K -1
        toggle = 0
        while k >= L:
            if ((k // L) %2) ==1:
                toggle ^=1
            k = k % L
        c = S[k]
        if toggle:
            c = c.swapcase()
        result.append(c)
    print(' '.join(result))

if __name__ == "__main__":
    main()