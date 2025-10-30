import threading
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N = int(input())
    P = list(map(int, input().split()))
    # termA: sum over inversions (i<j, P[i]>P[j]) of i
    # Use BIT over values to store counts and sum of positions
    class BIT:
        def __init__(self,n):
            self.n = n
            self.f = [0]*(n+1)
        def add(self,i,v):
            while i<=self.n:
                self.f[i]+=v
                i += i&-i
        def sum(self,i):
            s=0
            while i>0:
                s+=self.f[i]
                i -= i&-i
            return s
        def range_sum(self,l,r):
            return self.sum(r) - self.sum(l-1)
    # BIT over values 1..N
    bit_cnt = BIT(N)
    bit_pos = BIT(N)
    total_cnt = 0
    total_pos_sum = 0
    termA = 0
    for j, x in enumerate(P, start=1):
        # count of previous with value > x
        cnt_le = bit_cnt.sum(x)
        sumpos_le = bit_pos.sum(x)
        cnt_gt = total_cnt - cnt_le
        sumpos_gt = total_pos_sum - sumpos_le
        # add positions of those i < j with P[i]>P[j]
        termA += sumpos_gt
        # update
        bit_cnt.add(x, 1)
        bit_pos.add(x, j)
        total_cnt += 1
        total_pos_sum += j
    # termB: sum over inversions (i<j,P[i]>P[j]) of (# k in (i,j) with P[k]<P[j])
    # Computed by processing j in increasing P[j]
    # termB_j = sum_{k<j, P[k]<P[j]} [ (k-1) - (# processed < k) ]
    # = sum_k(k-1) - C(count_k,2), where count_k = #processed positions < j,
    # sum_k = sum of those k.
    bit_pcnt = BIT(N)
    bit_psum = BIT(N)
    termB = 0
    # prepare list of (value, position)
    vp = [(P[i], i+1) for i in range(N)]
    vp.sort()  # increasing value
    # processed positions = those with smaller values
    for val, j in vp:
        # count of processed positions < j
        ck = bit_pcnt.sum(j-1)
        sk = bit_psum.sum(j-1)
        # contribution: sum_k (k-1) - C(ck,2) = (sk - ck) - ck*(ck-1)/2
        termB += (sk - ck) - (ck * (ck - 1) // 2)
        # mark j
        bit_pcnt.add(j, 1)
        bit_psum.add(j, j)
    # answer
    print(termA + termB)

if __name__ == "__main__":
    main()