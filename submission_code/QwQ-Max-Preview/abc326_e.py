MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    if N == 0:
        print(0)
        return
    
    invN = pow(N, MOD-2, MOD)
    base = ( (N + 1) * invN ) % MOD
    
    res = 0
    for i in range(N):
        term = A[i] * pow(base, i, MOD) % MOD
        res = (res + term) % MOD
    
    res = res * invN % MOD
    print(res)
    
if __name__ == '__main__':
    main()