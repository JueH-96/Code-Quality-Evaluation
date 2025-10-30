class SegmentTree:
    def __init__(self, data):
        self.N = len(data)
        self.size = 1
        while self.size < self.N:
            self.size <<= 1
        self.tree = [(float('inf'), float('-inf'))] * (2 * self.size)
        for i in range(self.N):
            self.tree[self.size + i] = (data[i], data[i])
        for i in range(self.size - 1, 0, -1):
            left = self.tree[2 * i]
            right = self.tree[2 * i + 1]
            self.tree[i] = (min(left[0], right[0]), max(left[1], right[1]))

    def query(self, l, r):
        res_min = float('inf')
        res_max = float('-inf')
        l += self.size
        r += self.size
        while l < r:
            if l % 2 == 1:
                res_min = min(res_min, self.tree[l][0])
                res_max = max(res_max, self.tree[l][1])
                l += 1
            if r % 2 == 1:
                r -= 1
                res_min = min(res_min, self.tree[r][0])
                res_max = max(res_max, self.tree[r][1])
            l //= 2
            r //= 2
        return (res_min, res_max)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    pos = [0] * (N + 1)
    for i in range(N):
        pos[P[i]] = i + 1  # 1-based indexing for positions
    
    if K == 1:
        print(0)
        return
    
    segment_tree = SegmentTree(pos[1:N+1])
    
    min_diff = float('inf')
    for a in range(1, N - K + 2):
        min_pos, max_pos = segment_tree.query(a - 1, a + K - 1)
        diff = max_pos - min_pos
        if diff < min_diff:
            min_diff = diff
    print(min_diff)

if __name__ == "__main__":
    main()