import sys
sys.setrecursionlimit(10**7)
MOD = 998244353

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # Precompute factorials and inverses up to 3N say
    NX = 2* N + 5
    fact = [1]*(NX)
    invf = [1]*(NX)
    for i in range(1, NX):
        fact[i] = fact[i-1]*i % MOD
    invf[NX-1] = pow(fact[NX-1], MOD-2, MOD)
    for i in range(NX-1, 0, -1):
        invf[i-1] = invf[i]*i % MOD

    # A helper to form binomial(n,k)
    def comb(n,k):
        if k<0 or k>n: return 0
        return fact[n]*invf[k]%MOD*invf[n-k]%MOD

    # We'll keep a table Fsum[R][h] = sum_{t=0..h} F(R,t).
    # Actually we only need to build each R once, on demand, and then discard.
    # We'll do it in descending order of R, so we only carry one array in memory.
    Fsum = None
    lastR = -1

    ans = 0
    H = 0     # current prefix height = sum(A[j]-1) for j< i
    feasible = True

    for i in range(N):
        if not feasible: 
            break
        Ai = A[i]

        # compute R = number of remaining positions after i
        R = N - i - 1

        # If at any point H<0 or H> R+1, the given word A is no longer a valid
        # Łukasiewicz prefix, so we can't continue "equal" track beyond this.
        if H < 0 or H > R+1:
            feasible = False
            # but we do not 'break' immediately, we only stop adding
            # the final +1 if A itself were valid.
        else:
            # We want to add all ways that choose v < Ai here.
            # That runs v = 0..Ai-1, and new height h = H + (v-1),
            # so h runs from H-1 up to H + (Ai-1) - 1 = H+Ai-2.
            L = H - 1
            Rg = H + Ai - 2

            # Build (or reuse) the prefix sums Fsum[R].
            if lastR != R:
                # Build anew
                lastR = R
                Fsum = [0]*(R + N + 2)  # large enough
                running = 0
                # F(R,h) = (h+1)/(R+h+1) * binom(2R+h, R).
                # We'll do it for h=0..(R+max possible H).
                # But we only need up to h <= R+i_max, however
                # in worst‐case h ≤ (R+something). To stay safe we do up to N.
                for h in range(0, N+1):
                    num = comb(2*R + h, R)
                    val = num * (h+1) % MOD * pow(R+h+1, MOD-2, MOD) % MOD
                    running = (running + val) % MOD
                    Fsum[h] = running

            # Now pick off the range sum
            Lidx = max(L, 0)
            if Lidx <= Rg:
                tot = Fsum[min(Rg, N)]  # prefix up to Rg
                if Lidx > 0:
                    tot = (tot - Fsum[Lidx-1]) % MOD
                ans = (ans + tot) % MOD

        # Now update H as if we *chose* V_i = A_i
        H += (Ai - 1)

    # After the loop, if the entire A was *feasible* as a Polish word,
    # we must add +1 for the case "we never went < Ai at any i".
    if feasible and H == -1:
        ans = (ans + 1) % MOD

    print(ans)

if __name__ == "__main__":
    main()