import sys
input = sys.stdin.read

MOD = 998244353

def main():
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    sequences = []
    for _ in range(N):
        seq = list(map(int, data[idx:idx+M]))
        idx += M
        sequences.append(seq)
    
    result = 0
    
    for i in range(N):
        for j in range(i, N):
            if i == j:
                result = (result + 0) % MOD
                continue
            # Compute D = A_i XOR A_j
            D = [sequences[i][k] ^ sequences[j][k] for k in range(M)]
            found = False
            max_steps = 2 * M
            current = D.copy()
            for x in range(max_steps + 1):
                # Check if current is all zero
                is_zero = all(v == 0 for v in current)
                if is_zero:
                    result = (result + x) % MOD
                    found = True
                    break
                # Apply the operation once
                new_current = []
                total = 0
                for k in range(M):
                    total += current[k]
                    total %= 2
                    new_current.append(total)
                current = new_current
            if not found:
                result = (result + 0) % MOD
    
    print(result % MOD)

if __name__ == "__main__":
    main()