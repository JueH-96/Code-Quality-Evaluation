import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    MOD=998244353
    inv2=(MOD+1)//2
    N,K=map(int,sys.stdin.readline().split())
    P=list(map(int,sys.stdin.readline().split()))
    # compute initial inversions by BIT
    class BIT:
        def __init__(self,n):
            self.n=n
            self.f=[0]*(n+1)
        def add(self,i,v):
            while i<=self.n:
                self.f[i]+=v; i+=i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.f[i]; i-=i&-i
            return s
    bit=BIT(N)
    inv0=0
    for i,a in enumerate(P):
        # count how many >a before
        inv0 += i - bit.sum(a)
        bit.add(a,1)
    inv0 %= MOD

    # sum of expected inversions inside random K-block: each block gives K*(K-1)/4
    # there are (N-K+1) choices
    # so contributes K*(K-1)/4
    e_inblock = K*(K-1)%MOD * pow(4,MOD-2,MOD) %MOD

    # Now pairs crossing block boundary:
    # Precompute prefix counts of values <= v
    # We'll sweep segments and accumulate contribution
    # We maintain a BIT over values for current window
    bitw = BIT(N)
    # initial window [0..K-1]
    for i in range(K):
        bitw.add(P[i],1)
    cross = 0
    # Precompute prefix sums of P to help sums over outside positions
    # For each segment start s from 0 to N-K
    #   for each outside x<s: contributes sum_less(P[x])/K
    #   for each outside x>R: contributes sum_greater(P[x])/K
    # We can maintain two BITs for outside-left and outside-right
    bitR = BIT(N)
    for i in range(K, N):
        bitR.add(P[i],1)
    bitL = BIT(N)
    for s in range(0, N-K+1):
        # window is [s..s+K-1], window BIT=bitw
        # left outside positions [0..s-1] in bitL
        # right outside positions [s+K..N-1] in bitR
        # for each left x: probability P[x] > random from window = count_less_window(P[x])/K
        # sum over x left of count_less_window(P[x])
        # = sum over v in window of (# left with P[x]>v)
        #   = sum_{v in window} (bitL.sum(N)-bitL.sum(v))
        # similarly right x: probability random_window > P[x] = count_greater_window(P[x])/K
        # sum = sum_{v in window}(bitR.sum(v-1))
        # compute sums by iterating over window values? too big.
        # But we can maintain window sum of bitL.sum(N)-bitL.sum(v) by keeping track of window elements
        # So we precompute for window sumL = sum(bitL.sum(N)-bitL.sum(v)) over v in window
        # and sumR = sum(bitR.sum(v-1)) over v in window
        # We'll update these incrementally.
        # On first s=0, compute both
        if s==0:
            sumL = 0
            sumR = 0
            # iterate window
            for i in range(s, s+K):
                v = P[i]
                sumL = (sumL + (bitL.sum(N) - bitL.sum(v)))%MOD
                sumR = (sumR + bitR.sum(v-1))%MOD
        cross = (cross + sumL + sumR)%MOD
        # slide window: remove P[s], add P[s+K]
        if s+K < N:
            v_out = P[s]
            # remove v_out from window, add to left
            # update sumL and sumR for new window
            # removing v_out: subtract its contributions
            sumL = (sumL - (bitL.sum(N) - bitL.sum(v_out)))%MOD
            sumR = (sumR - bitR.sum(v_out-1))%MOD
            bitL.add(v_out,1)
            bitw.add(v_out,-1)
            # add v_in
            v_in = P[s+K]
            bitR.add(v_in,-1)
            bitw.add(v_in,1)
            sumL = (sumL + (bitL.sum(N) - bitL.sum(v_in)))%MOD
            sumR = (sumR + bitR.sum(v_in-1))%MOD
    inv_cross = cross * pow(K, MOD-2, MOD) %MOD * pow(N-K+1, MOD-2, MOD) %MOD

    ans = (inv0 + e_inblock + inv_cross) %MOD
    print(ans)

if __name__=='__main__':
    main()