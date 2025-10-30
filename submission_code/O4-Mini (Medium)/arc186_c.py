import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1<<25)
    input = sys.stdin.readline

    T = int(input())
    allN = 0
    # We'll process each test independently
    for _ in range(T):
        N,M = map(int, input().split())
        allN += N
        boxes = [tuple(map(int, input().split())) for __ in range(N)]
        # boxes = list of (V_i,P_i)

        # Sort boxes by price ascending; remember original index
        boxesP = sorted((P,V,i) for i,(V,P) in enumerate(boxes))
        # We'll refer to them by their position in boxesP[0..N-1]
        #   boxesP[pos] = (P_i,V_i,orig_i)
        P_list = [p for p,v,i in boxesP]

        # Build a list of (V, pos_in_P_list), sorted descending in V
        byV = []
        for pos,(p,v,ii) in enumerate(boxesP):
            byV.append((v,pos))
        byV.sort(reverse=True)

        # All distinct V values, descending
        uniqV = sorted({v for v,p in boxes}, reverse=True)

        # Fenwick on counts and on sum of P
        class Fenwick:
            def __init__(self,n):
                self.n=n
                self.c1=[0]*(n+1)
                self.c2=[0]*(n+1)
            def add(self,idx,delta_cnt,delta_sum):
                i=idx+1
                while i<=self.n:
                    self.c1[i]+=delta_cnt
                    self.c2[i]+=delta_sum
                    i+=i&-i
            def sum(self,idx):
                "sum over [0..idx]"
                i=idx+1
                s1=0; s2=0
                while i>0:
                    s1+=self.c1[i]
                    s2+=self.c2[i]
                    i-=i&-i
                return s1,s2
            def find_kth(self,k):
                "smallest idx so that prefix_count >= k"
                # if total < k, returns n
                idx=0
                bit=1<<(self.n.bit_length())
                cur=0
                while bit>0:
                    nxt=idx+bit
                    if nxt<=self.n and self.c1[nxt]<k:
                        k-=self.c1[nxt]
                        idx=nxt
                    bit>>=1
                return idx  # this is 1-based sum index – returns idx so that sum up to idx has at least original k
        fw = Fenwick(N)

        ans = 0
        ptr = 0  # pointer in byV
        # We sweep J descending
        import bisect
        for J in uniqV:
            # activate all boxes with V >= J
            while ptr < N and byV[ptr][0] >= J:
                _v,pos = byV[ptr]
                # add this box at position pos, count=1, sum=P_list[pos]
                fw.add(pos,1,P_list[pos])
                ptr+=1
            # among those inserted, we only want those with P < J
            # find boundary in P_list: posJ = first pos where P_list[pos]>=J
            posJ = bisect.bisect_left(P_list, J)
            cnt, sP = fw.sum(posJ-1) if posJ>0 else (0,0)
            if cnt==0:
                continue
            k = min(cnt, M)
            if cnt <= M:
                # all boxes with P<J count
                total_gain = J*cnt - sP
            else:
                # need sum of smallest M P's among them
                # find t so that prefix count up to t >= M
                t = fw.find_kth(M)
                # because find_kth gives 1-based Fenw idx, we do prefix sum up to t-1 in zero-based
                csum, ssum = fw.sum(t-1)
                # that ssum is sum of exactly M prices
                total_gain = J*M - ssum
            if total_gain>ans:
                ans = total_gain

        if ans<0:
            ans = 0
        print(ans)

if __name__=="__main__":
    main()