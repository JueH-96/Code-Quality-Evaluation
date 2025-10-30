MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    sequences = []
    for _ in range(N):
        seq = list(map(int, input[ptr:ptr+M]))
        ptr += M
        sequences.append(seq)
    
    total = 0
    for i in range(N):
        for j in range(i, N):
            if i == j:
                continue
            D = [sequences[i][k] ^ sequences[j][k] for k in range(M)]
            found = False
            for x in range(0, 20):
                current = D.copy()
                for step in range(x):
                    new_current = []
                    for k in range(M):
                        s = sum(current[:k+1]) % 2
                        new_current.append(s)
                    current = new_current
                if all(bit == 0 for bit in current):
                    total += x
                    found = True
                    break
            if found:
                continue
    print(total % MOD)

if __name__ == '__main__':
    main()