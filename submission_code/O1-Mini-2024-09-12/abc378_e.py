# YOUR CODE HERE
import sys

class Fenwick:
    def __init__(self, size):
        self.size = size
        self.tree = [0]*(size +2)
    
    def update(self, idx, delta=1):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        res = 0
        while idx >0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    fenwick = Fenwick(M)
    s_prev = 0
    fenwick.update(s_prev +1 )  # s0=0
    sum_s =0
    res =0
    for i in range(N):
        s_r = (s_prev + A[i]) % M
        # s_r ranges from 0 to M-1, map to 1 to M
        C = fenwick.query(s_r +1 )
        H = (i +1) - C
        term = (i +1)*s_r - sum_s + H * M
        res += term
        sum_s += s_r
        fenwick.update(s_r +1 )
        s_prev = s_r
    print(res)

if __name__ == "__main__":
    main()