import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, W = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    NEG_INF = -10**30

    # dp[w] = max happiness with total weight w
    dp = [NEG_INF] * (W + 1)
    dp[0] = 0

    # Convex hull trick for max queries, slopes increasing, queries x increasing
    class CHT:
        __slots__ = ('a', 'b', 'dq')
        def __init__(self):
            # lines are (a, b) meaning y = a*x + b
            self.dq = []
        def bad(self, l1, l2, l3):
            # return True if l2 is unnecessary between l1 and l3
            # intersection(l1,l2) >= intersection(l2,l3)
            # (b2-b1)/(a1-a2) >= (b3-b2)/(a2-a3)
            # cross-multiplied:
            # (b2-b1)*(a2-a3) >= (b3-b2)*(a1-a2)
            return (l2[1]-l1[1])*(l2[0]-l3[0]) >= (l3[1]-l2[1])*(l1[0]-l2[0])
        def add(self, a, b):
            # add line y = a*x + b
            dq = self.dq
            dq.append((a,b))
            # maintain convexity
            while len(dq) >= 3 and self.bad(dq[-3], dq[-2], dq[-1]):
                # remove middle
                dq.pop(-2)
        def query(self, x):
            dq = self.dq
            # pop front if next is better
            while len(dq) >= 2 and dq[0][0]*x + dq[0][1] <= dq[1][0]*x + dq[1][1]:
                dq.pop(0)
            return dq[0][0]*x + dq[0][1]

    # Process each item type as a group-knapsack with concave profits
    for w_i, v_i in items:
        newdp = [dp[w] for w in range(W+1)]
        # For each residue class modulo w_i
        for r in range(w_i):
            # collect dp_old at weights r + k*w_i
            arr = []
            pos = r
            while pos <= W:
                arr.append(dp[pos])
                pos += w_i
            L = len(arr)
            if L == 0:
                continue
            # Build lines from k=0..L-1
            # B[k] = arr[k] - k^2 - k*v_i
            cht = CHT()
            for k in range(L):
                Bk = arr[k] - k*k - k*v_i
                # line slope = 2*k, intercept = Bk
                cht.add(2*k, Bk)
            # now query for each m=0..L-1
            pos = r
            for m in range(L):
                # dp_new at weight = r + m*w_i
                best = cht.query(m)
                # dp_new = m*v_i - m^2 + best
                val = m*v_i - m*m + best
                if val > newdp[pos]:
                    newdp[pos] = val
                pos += w_i
        dp = newdp

    # answer is max over dp[0..W]
    print(max(dp))

if __name__ == "__main__":
    main()