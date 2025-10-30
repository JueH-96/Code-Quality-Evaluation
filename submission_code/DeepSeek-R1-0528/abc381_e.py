import sys

INF = 10**18
NEG_INF = -10**18

class SegmentTreeMax:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [NEG_INF] * (2 * self.size)
    
    def update(self, pos, val):
        idx = pos - 1 + self.size
        self.data[idx] = val
        while idx > 1:
            idx //= 2
            self.data[idx] = max(self.data[2*idx], self.data[2*idx+1])
    
    def query(self, l, r):
        l0 = l - 1
        r0 = r - 1
        l0 += self.size
        r0 += self.size
        res = NEG_INF
        while l0 <= r0:
            if l0 % 2 == 1:
                res = max(res, self.data[l0])
                l0 += 1
            if r0 % 2 == 0:
                res = max(res, self.data[r0])
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

class SegmentTreeMin:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [INF] * (2 * self.size)
    
    def update(self, pos, val):
        idx = pos - 1 + self.size
        self.data[idx] = val
        while idx > 1:
            idx //= 2
            self.data[idx] = min(self.data[2*idx], self.data[2*idx+1])
    
    def query(self, l, r):
        l0 = l - 1
        r0 = r - 1
        l0 += self.size
        r0 += self.size
        res = INF
        while l0 <= r0:
            if l0 % 2 == 1:
                res = min(res, self.data[l0])
                l0 += 1
            if r0 % 2 == 0:
                res = min(res, self.data[r0])
                r0 -= 1
            l0 //= 2
            r0 //= 2
        return res

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    first_line = data[0].split()
    N = int(first_line[0])
    Q = int(first_line[1])
    S = data[1].strip()
    
    prefix1 = [0] * (N+1)
    prefix2 = [0] * (N+1)
    for i in range(1, N+1):
        ch = S[i-1]
        prefix1[i] = prefix1[i-1] + (1 if ch == '1' else 0)
        prefix2[i] = prefix2[i-1] + (1 if ch == '2' else 0)
    
    centers = []
    for i in range(1, N+1):
        if S[i-1] == '/':
            x = prefix1[i-1]
            y = prefix2[i]
            v = x + y
            centers.append((v, i, x, y))
    
    queries = []
    index = 2
    for i in range(Q):
        parts = data[index].split()
        index += 1
        if not parts:
            continue
        L = int(parts[0]); R = int(parts[1])
        queries.append((L, R))
    
    centers1 = sorted(centers, key=lambda x: x[0])
    queries1 = []
    for idx, (L, R) in enumerate(queries):
        c = prefix1[L-1]
        d = prefix2[R]
        T_val = c + d
        queries1.append((T_val, L, R, idx))
    queries1_sorted = sorted(queries1, key=lambda x: x[0])
    
    seg_max = SegmentTreeMax(N)
    pointer = 0
    res1 = [NEG_INF] * len(queries)
    
    for T_val, L_val, R_val, idx in queries1_sorted:
        while pointer < len(centers1) and centers1[pointer][0] <= T_val:
            v, pos, x, y = centers1[pointer]
            seg_max.update(pos, x)
            pointer += 1
        max_x = seg_max.query(L_val, R_val)
        res1[idx] = max_x
        
    centers2 = sorted(centers, key=lambda x: x[0], reverse=True)
    queries2 = []
    for idx, (L, R) in enumerate(queries):
        c = prefix1[L-1]
        d = prefix2[R]
        T_val = c + d
        queries2.append((T_val, L, R, idx))
    queries2_sorted = sorted(queries2, key=lambda x: x[0], reverse=True)
    
    seg_min = SegmentTreeMin(N)
    pointer = 0
    res2 = [INF] * len(queries)
    
    for T_val, L_val, R_val, idx in queries2_sorted:
        while pointer < len(centers2) and centers2[pointer][0] > T_val:
            v, pos, x, y = centers2[pointer]
            seg_min.update(pos, y)
            pointer += 1
        min_y = seg_min.query(L_val, R_val)
        res2[idx] = min_y
        
    for idx in range(len(queries)):
        L, R = queries[idx]
        c = prefix1[L-1]
        d = prefix2[R]
        cand1 = NEG_INF
        if res1[idx] != NEG_INF:
            cand1 = 2 * (res1[idx] - c) + 1
        cand2 = NEG_INF
        if res2[idx] != INF:
            cand2 = 2 * (d - res2[idx]) + 1
        ans = max(0, cand1, cand2)
        print(ans)

if __name__ == "__main__":
    main()