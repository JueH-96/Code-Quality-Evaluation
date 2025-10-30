import sys
import threading
def main():
    import sys
    MOD = 998244353
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    # Initial X[i] = i%2
    if A[0] != 1 or A[-1] != N%2:
        print(0); return
    # Build runs of A
    runs = []
    prev = A[0]; cnt = 1
    for x in A[1:]:
        if x==prev:
            cnt+=1
        else:
            runs.append((prev,cnt))
            prev=x; cnt=1
    runs.append((prev,cnt))
    M = len(runs)
    # Precompute factorials for Catalan: C_{n} = comb(2n,n)/(n+1)
    maxk = N
    fact = [1]*(2*maxk+1)
    invf = [1]*(2*maxk+1)
    for i in range(1,2*maxk+1):
        fact[i] = fact[i-1]*i % MOD
    invf[2*maxk] = pow(fact[2*maxk], MOD-2, MOD)
    for i in range(2*maxk,0,-1):
        invf[i-1] = invf[i]*i % MOD
    def comb(n,r):
        if r<0 or r>n: return 0
        return fact[n]*invf[r]%MOD*invf[n-r]%MOD
    def catalan(n):
        if n<=0: return 1
        # C_n = comb(2n,n) * inv(n+1)
        return comb(2*n,n)*pow(n+1, MOD-2, MOD) % MOD

    ans = 1
    pos = 1
    for color, length in runs:
        # segment from pos..pos+length-1
        s = pos
        L = length
        # count initial runs of this color in [s..s+L-1]
        p = 1 if (s%2)==color else 0
        k = (L + p)//2
        # need to merge k initial runs into 1 -> k-1 merges -> Catalan[k-1]
        ans = ans * catalan(k-1) % MOD
        pos += L
    print(ans)

if __name__=='__main__':
    main()