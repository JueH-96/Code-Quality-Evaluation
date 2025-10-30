import sys
import math
import numpy as np

def main():
    import sys
    import math
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx]); idx +=1
    D = list(map(int, data[idx:idx+N])); idx +=N
    L1, C1, K1 = map(int, data[idx:idx+3]); idx +=3
    L2, C2, K2 = map(int, data[idx:idx+3]); idx +=3
    
    INF = 10**18
    DP = np.full((K1+1, K2+1), INF, dtype=np.int64)
    DP[0,0] = 0
    
    for d in D:
        options = []
        max_x = min(K1, (d + L1 -1)//L1)
        for x in range(0, max_x +1):
            remain = d - x * L1
            if remain <=0:
                y =0
            else:
                y = (remain + L2 -1) // L2
            if y <= K2:
                cost = x * C1 + y * C2
                options.append( (x, y, cost) )
        if not options:
            print(-1)
            return
        new_DP = np.full((K1+1, K2+1), INF, dtype=np.int64)
        for dx, dy, c_add in options:
            if dx > K1 or dy > K2:
                continue
            shifted_x = K1 +1 - dx
            shifted_y = K2 +1 - dy
            if shifted_x <=0 or shifted_y <=0:
                continue
            shifted = DP[:shifted_x, :shifted_y] + c_add
            np.minimum(new_DP[dx:, dy:], shifted, out=new_DP[dx:, dy:])
        DP = new_DP
    min_cost = DP[:K1+1, :K2+1].min()
    if min_cost < INF:
        print(min_cost)
    else:
        print(-1)
            
if __name__ == "__main__":
    main()