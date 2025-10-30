class Fenwick:
    def __init__(self, size):
        self.N = size + 5
        self.bit = [0] * self.N
    
    def query(self, idx):
        ans = 0
        while idx > 0:
            ans ^= self.bit[idx]
            idx -= idx & -idx
        return ans
    
    def update(self, idx, val):
        while idx < self.N:
            self.bit[idx] ^= val
            idx += idx & -idx

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    queries = data[3:]
    
    same = [0] * (N - 1)
    for i in range(N - 1):
        same[i] = 1 if S[i] == S[i+1] else 0
    prefix_same = [0] * N
    for i in range(1, N):
        prefix_same[i] = prefix_same[i-1] + same[i-1]
    
    ft = Fenwick(N)
    
    idx = 0
    for _ in range(Q):
        t = int(queries[idx])
        if t == 1:
            L = int(queries[idx+1])
            R = int(queries[idx+2])
            ft.update(L, 1)
            if R + 1 <= N:
                ft.update(R + 1, 1)
        else:
            L = int(queries[idx+1])
            R = int(queries[idx+2])
            if L > 1:
                sum_same = prefix_same[R-1] - prefix_same[L-2]
            else:
                sum_same = prefix_same[R-1]
            flip_L = ft.query(L)
            flip_R = ft.query(R)
            sum_flip_diff = flip_L ^ flip_R
            if (sum_same ^ sum_flip_diff) == 0:
                print("Yes")
            else:
                print("No")
        idx += 3

if __name__ == "__main__":
    main()