import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = list(map(int, input[ptr:ptr+N]))
    
    # Create Q_inv: Q_inv[v] = i where Q[i-1] = v (since original indices are 1-based in the problem)
    Q_inv = [0] * (N + 1)
    for i in range(N):
        val = Q[i]
        Q_inv[val] = i + 1  # because original i is 1-based
    
    S = []
    for bib_i in range(1, N+1):
        original_i = Q_inv[bib_i]
        stared_at_original_i = P[original_i - 1]  # P is 1-based
        stared_at_bib = Q[stared_at_original_i - 1]
        S.append(str(stared_at_bib))
    
    print(' '.join(S))

if __name__ == '__main__':
    main()