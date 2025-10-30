import sys
import threading
def main():
    import sys
    input = sys.stdin.readline

    N, K, Q = map(int, input().split())
    X = [0]*Q
    Y = [0]*Q
    vals = {0}
    for i in range(Q):
        xi, yi = map(int, input().split())
        X[i] = xi-1
        Y[i] = yi
        vals.add(yi)
    # coordinate compression
    comp = {v:i+1 for i,v in enumerate(sorted(vals))}
    M = len(comp)

    class BIT:
        def __init__(self,n):
            self.n=n
            self.t=[0]*(n+1)
        def add(self,i,v):
            # i in [1..n]
            while i<=self.n:
                self.t[i]+=v
                i+=i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.t[i]
                i-=i&-i
            return s
        # find largest idx such that prefix sum <= x
        def find_le(self,x):
            if x < 0:
                return 0
            pos=0
            bit = 1<<(self.n.bit_length())
            s=0
            while bit:
                nxt = pos + bit
                if nxt <= self.n and s + self.t[nxt] <= x:
                    s += self.t[nxt]
                    pos = nxt
                bit >>= 1
            return pos

    bit_cnt = BIT(M)
    bit_sum = BIT(M)
    # initial all zeros
    zero_idx = comp[0]
    bit_cnt.add(zero_idx, N)
    # sum of zeros adds nothing to bit_sum

    A = [0]*N
    total_sum = 0

    out = []
    for i in range(Q):
        pos = X[i]
        newv = Y[i]
        oldv = A[pos]
        if oldv == newv:
            # no change
            # compute f(A) below
            pass
        else:
            # remove old
            oi = comp[oldv]
            bit_cnt.add(oi, -1)
            bit_sum.add(oi, -oldv)
            total_sum -= oldv
            # add new
            ni = comp[newv]
            bit_cnt.add(ni, 1)
            bit_sum.add(ni, newv)
            total_sum += newv
            A[pos] = newv

        # compute sum of top K
        # find threshold index
        # want minimal idx such that count[idx..M] >= K
        # let x = N-K, find p = largest prefix_count <= x
        p = bit_cnt.find_le(N - K)
        thr = p+1
        if thr > M:
            out.append("0")
        else:
            # sum of greater = total_sum - prefix_sum up to p
            sum_pref = bit_sum.sum(p)
            cnt_pref = bit_cnt.sum(p)
            sum_greater = total_sum - sum_pref
            cnt_greater = N - cnt_pref
            need = K - cnt_greater
            # value at thr
            v_thr = sorted(vals)[thr-1]  # but this is O(1)? Actually this is O(1) if we store reverse map
            # To avoid re-sorting, build an array:
            # We'll build arr_comp so that arr_comp[i] = original value of coord i
            # Let's build it outside.

            # For now workaround: we build once:
            # Actually move construction out of loop.

            # But here we assume we have arr_comp:

            res = sum_greater + need * arr_comp[thr]
            out.append(str(res))

    sys.stdout.write("
".join(out))

# Fix array for reverse comp
def _bootstrap():
    import sys
    input = sys.stdin.readline

    N, K, Q = map(int, input().split())
    X = [0]*Q
    Y = [0]*Q
    vals = {0}
    for i in range(Q):
        xi, yi = map(int, input().split())
        X[i] = xi-1
        Y[i] = yi
        vals.add(yi)
    sorted_vals = sorted(vals)
    comp = {v:i+1 for i,v in enumerate(sorted_vals)}
    global arr_comp
    arr_comp = [0]*(len(sorted_vals)+1)
    for i,v in enumerate(sorted_vals, start=1):
        arr_comp[i] = v

    class BIT:
        def __init__(self,n):
            self.n=n
            self.t=[0]*(n+1)
        def add(self,i,v):
            while i<=self.n:
                self.t[i]+=v
                i+=i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.t[i]
                i-=i&-i
            return s
        def find_le(self,x):
            if x<0:
                return 0
            pos=0
            bit=1<<(self.n.bit_length())
            s=0
            while bit:
                nxt=pos+bit
                if nxt<=self.n and s+self.t[nxt]<=x:
                    s+=self.t[nxt]
                    pos=nxt
                bit>>=1
            return pos

    M = len(sorted_vals)
    bit_cnt = BIT(M)
    bit_sum = BIT(M)
    zero_idx = comp[0]
    bit_cnt.add(zero_idx, N)
    A = [0]*N
    total_sum = 0

    out = []
    for i in range(Q):
        pos = X[i]
        newv = Y[i]
        oldv = A[pos]
        if oldv != newv:
            oi = comp[oldv]; ni = comp[newv]
            bit_cnt.add(oi, -1)
            bit_sum.add(oi, -oldv)
            total_sum -= oldv
            bit_cnt.add(ni, 1)
            bit_sum.add(ni, newv)
            total_sum += newv
            A[pos] = newv

        p = bit_cnt.find_le(N - K)
        thr = p+1
        if thr > M:
            out.append("0")
        else:
            sum_pref = bit_sum.sum(p)
            cnt_pref = bit_cnt.sum(p)
            sum_greater = total_sum - sum_pref
            cnt_greater = N - cnt_pref
            need = K - cnt_greater
            res = sum_greater + need * arr_comp[thr]
            out.append(str(res))

    sys.stdout.write("
".join(out))

# we use the combined bootstrap as main
if __name__ == "__main__":
    _bootstrap()