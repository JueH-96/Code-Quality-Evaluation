MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = []
    for _ in range(N):
        row = list(map(int, input[ptr:ptr+M]))
        ptr += M
        A.append(row)
    
    # Precompute the order of T for M
    order = 1
    while order < M:
        order <<= 1
    
    total = 0
    for i in range(N):
        for j in range(i, N):
            D = [ (A[i][k] ^ A[j][k]) for k in range(M) ]
            # Check if D is all zero
            if all(x == 0 for x in D):
                total += 0
                continue
            # Simulate up to order steps
            current = D.copy()
            found = False
            for x in range(0, order + 1):
                if all(x == 0 for x in current):
                    total += x
                    found = True
                    break
                # Apply transformation
                new_current = []
                s = 0
                for k in range(M):
                    s += current[k]
                    s %= 2
                    new_current.append(s)
                current = new_current
            if not found:
                total += 0
    print(total % MOD)

main()