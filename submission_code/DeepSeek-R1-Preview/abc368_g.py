import sys
import math

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())
    queries = [sys.stdin.readline().strip() for _ in range(Q)]
    
    # Fenwick Tree for sum of A
    class FenwickSum:
        def __init__(self, size):
            self.n = size
            self.tree = [0] * (self.n + 1)
        
        def update(self, idx, delta):
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Fenwick Tree for sum of logs of B
    class FenwickLog:
        def __init__(self, size):
            self.n = size
            self.tree = [0.0] * (self.n + 1)
            self.current = [0.0] * (self.n + 1)
        
        def update(self, idx, new_val):
            old_val = self.current[idx]
            delta = new_val - old_val
            self.current[idx] = new_val
            while idx <= self.n:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0.0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    # Initialize Fenwick trees
    fen_sum_a = FenwickSum(N)
    for i in range(1, N+1):
        fen_sum_a.update(i, A[i-1])
    
    fen_log_b = FenwickLog(N)
    log_B = [0.0] * (N + 1)
    for i in range(1, N+1):
        log_B[i] = math.log(B[i-1])
        fen_log_b.update(i, log_B[i])
    
    # Process queries
    for q in queries:
        parts = q.split()
        if not parts:
            continue
        if parts[0] == '1':
            # Type 1: update A
            i = int(parts[1])
            x = int(parts[2])
            delta = x - A[i-1]
            A[i-1] = x
            fen_sum_a.update(i, delta)
        elif parts[0] == '2':
            # Type 2: update B
            i = int(parts[1])
            x = int(parts[2])
            old_log = log_B[i]
            new_log = math.log(x)
            log_B[i] = new_log
            fen_log_b.update(i, new_log)
            B[i-1] = x
        else:
            # Type 3: query [l, r]
            l = int(parts[1])
            r = int(parts[2])
            if l == r:
                # Only one element, must add
                print(A[l-1])
                continue
            # Simulate the process
            v = A[l-1]
            for i in range(l, r):
                current_i = i + 1  # since l is the first, next is l+1
                a = A[current_i -1]
                b = B[current_i -1]
                if b == 1:
                    v += a
                    continue
                t_i = a / (b - 1)
                if v < t_i:
                    v += a
                else:
                    v *= b
            print(v)

if __name__ == '__main__':
    main()