# YOUR CODE HERE
import sys
import sys
import sys
import sys
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    from bisect import bisect_right
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    H = list(map(int, data[2:2+N]))
    queries = []
    for i in range(Q):
        l = int(data[2+N+2*i])
        r = int(data[2+N+2*i+1])
        queries.append( (l, r, i) )
    
    # Step 1: Compute P[x] using monotonic stack
    P = [0] * (N+1)  # 1-based indexing
    stack = []
    for x in range(1, N+1):
        while stack and H[stack[-1]-1] < H[x-1]:
            stack.pop()
        if stack:
            P[x] = stack[-1]
        else:
            P[x] = 0
        stack.append(x)
    
    # Step 2: Sort queries by l in ascending order
    queries_sorted = sorted(queries, key=lambda x: x[0])
    
    # Step 3: Sort x by P[x] in ascending order
    x_sorted = sorted(range(1, N+1), key=lambda x: P[x])
    
    # Step 4: Initialize BIT
    class BIT:
        def __init__(self, size):
            self.N = size
            self.tree = [0] * (self.N + 2)
        
        def update(self, idx, delta=1):
            while idx <= self.N:
                self.tree[idx] += delta
                idx += idx & -idx
        
        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res
    
    bit = BIT(N)
    
    # Step 5: Process queries
    answers = [0] * Q
    ptr = 0
    for l, r, idx in queries_sorted:
        # Insert all x with P[x] < l
        while ptr < N and P[x_sorted[ptr]] < l:
            bit.update(x_sorted[ptr])
            ptr +=1
        # Now, number of x > r with P[x] < l is bit.query(N) - bit.query(r)
        count = bit.query(N) - bit.query(r)
        answers[idx] = count
    
    # Step 6: Output
    for ans in answers:
        print(ans)

if __name__ == "__main__":
    main()