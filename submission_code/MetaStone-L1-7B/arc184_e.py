import sys

MOD = 10**9 + 9

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    A = []
    for _ in range(N):
        row = list(map(int, data[idx:idx+M]))
        idx += M
        A.append(row)
    
    total = 0
    
    for i in range(N):
        for j in range(N):
            if i == j:
                total += 0
                continue
            
            D = [0] * M
            for k in range(M):
                D[k] = (A[i][k] - A[j][k]) % 2
            
            if all(d == 0 for d in D):
                total += 0
                continue
            
            found = False
            for x in range(1, M+1):
                new_D = D.copy()
                for k in range(M):
                    s = sum(new_D[:k+1]) % 2
                    new_D[k] = s
                if all(d == 0 for d in new_D):
                    total += x
                    found = True
                    break
            if not found:
                total += 0
    
    print(total % MOD)

if __name__ == '__main__':
    main()