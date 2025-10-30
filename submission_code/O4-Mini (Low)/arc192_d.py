import sys
import threading
MOD = 998244353

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # Precompute primes up to 1000
    mx = 1000
    spf = list(range(mx+1))
    for i in range(2, int(mx**0.5)+1):
        if spf[i]==i:
            for j in range(i*i, mx+1, i):
                if spf[j]==j:
                    spf[j]=i

    # Factor each A[i] into primes
    # Exponents[i] = dict p -> a_{i,p}
    exponents = [dict() for _ in range(N-1)]
    for i in range(N-1):
        x = A[i]
        while x>1:
            p = spf[x]
            cnt = 0
            while x%p==0:
                x//=p
                cnt+=1
            exponents[i][p] = cnt

    # For each prime p that appears anywhere, do a DP
    result = 1
    for p in range(2, mx+1):
        # gather the exponent pattern for this p
        a = [ exponents[i].get(p, 0) for i in range(N-1) ]
        if all(v==0 for v in a):
            continue  # prime p does not appear

        # dp[i][e_left][e_right][have_zero]
        # = number of ways to pick directions for edges 1..i
        #   so that at node i we have exponent e_left,
        #                at node i+1 we have exponent e_right,
        #   and have_zero=0/1 indicates whether among finalized nodes
        #   (1..i-1) we've already seen e_j=0 for some j.
        #
        # e_left, e_right range 0..sum(a)
        S = sum(a)
        dp = [ [ [0]*2 for _ in range(S+1)] for __ in range(S+1) ]
        # Initialization at i=0: no edges chosen,
        # effectively there's only node 1 so far with exp=0,
        # and node 2 not yet started.  We model that as
        # dp[e1=0][e2=0][have_zero=1] = 1
        # (we already have a zero among finalized, i.e. node 1).
        dp[0][0][1] = 1

        for idx in range(N-1):
            ai = a[idx]
            nxt = [ [ [0]*2 for _ in range(S+1)] for __ in range(S+1) ]
            for e1 in range(S+1):
                for e2 in range(S+1):
                    for hz in (0,1):
                        val = dp[e1][e2][hz]
                        if not val: continue
                        # Option +: we add ai to e1
                        ne1, ne2 = e1+ai, e2
                        nh = hz or (e1==0)  # once we finalize node idx+1?
                        # Actually node idx+1 becomes 'final' for e1 -> check zero there
                        nh |= (ne1==0)
                        if ne1<=S:
                            nxt[ne1][ne2][nh] = (nxt[ne1][ne2][nh] + val) % MOD

                        # Option -: we add ai to e2
                        ne1, ne2 = e1, e2+ai
                        nh = hz or (e2==0)
                        if ne2<=S:
                            nxt[ne1][ne2][nh] = (nxt[ne1][ne2][nh] + val) % MOD

            dp = nxt

        # After all edges, nodes N-1,N exponents are in dp[..]
        # We must also enforce gcd=1 => have seen a zero somewhere among all N nodes.
        ways = 0
        for e1 in range(S+1):
            for e2 in range(S+1):
                ways = (ways + dp[e1][e2][1]) % MOD

        # Next: each valid pattern contributes to the TOTAL *p*-exponent
        # of the product of all S_j.  That exponent is
        # sum_{node j=1..N} e_{j,p}.  But by construction
        # that sum = sum(a_i) no matter how we assign directions.
        # Hence every of these 'ways' patterns contributes p^{sum(a)}.
        result = result * pow(p, S, MOD) % MOD
        result = result * ways % MOD

    print(result)

if __name__=="__main__":
    main()