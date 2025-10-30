import sys
from collections import defaultdict

MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    A = []
    for _ in range(N):
        row = list(map(int, input[idx:idx+M]))
        idx += M
        A.append(row)
    
    # Precompute up to 20 transformations for each sequence
    MAX_X = 20
    transformed = []
    for seq in A:
        t = [seq.copy()]
        for _ in range(MAX_X):
            new_seq = []
            s = 0
            for val in t[-1]:
                s = (s + val) % 2
                new_seq.append(s)
            t.append(new_seq)
        transformed.append(t)
    
    result = 0
    for i in range(N):
        for j in range(i, N):
            found = False
            for x in range(MAX_X + 1):
                # Compare transformed[i][x] and transformed[j][x]
                ti = transformed[i][x]
                tj = transformed[j][x]
                equal = True
                for a, b in zip(ti, tj):
                    if a != b:
                        equal = False
                        break
                if equal:
                    result = (result + x) % MOD
                    found = True
                    break
            if not found:
                result = (result + 0) % MOD
    print(result)

if __name__ == '__main__':
    main()