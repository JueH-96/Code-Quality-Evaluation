import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    
    class FenwickTree:
        def __init__(self, size):
            self.size = size
            self.tree = [0] * (self.size + 2)  # 1-based indexing
        
        def update(self, idx, delta):
            idx += 1  # Convert to 1-based index
            while idx <= self.size:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            idx += 1  # Convert to 1-based index
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    ft = FenwickTree(M)
    P_prev = 0
    ft.update(P_prev, 1)
    sum_P_prev = 0
    ans = 0
    
    for idx in range(N):
        a = A[idx]
        P_next = (P_prev + a) % M
        cnt = ft.query(M - 1) - ft.query(P_next)
        current_r = idx + 1
        sum_S_i = P_next * current_r - sum_P_prev + M * cnt
        ans += sum_S_i
        sum_P_prev += P_next
        ft.update(P_next, 1)
        P_prev = P_next
    
    print(ans)

if __name__ == "__main__":
    main()