import sys
import bisect

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    H = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Compute PGE for 1-based indices
    PGE = [0] * (N + 2)  # 1-based to N
    stack = []
    for j in range(1, N+1):
        while stack and H[stack[-1]-1] < H[j-1]:
            stack.pop()
        if stack:
            PGE[j] = stack[-1]
        else:
            PGE[j] = 0
        stack.append(j)
    
    # Prepare data array (0-based)
    data = [0] * N
    for i in range(N):
        data[i] = PGE[i+1]  # j = i+1 (1-based) -> data[i]

    # Segment Tree implementation
    class SegmentTree:
        def __init__(self, array):
            self.n = len(array)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.tree = [[] for _ in range(2 * self.size)]
            # Fill leaves
            for i in range(self.n):
                self.tree[self.size + i] = [array[i]]
            # Build parents
            for v in range(self.size - 1, 0, -1):
                # Merge left and right children
                left = self.tree[2 * v]
                right = self.tree[2 * v + 1]
                merged = []
                i = j = 0
                while i < len(left) and j < len(right):
                    if left[i] < right[j]:
                        merged.append(left[i])
                        i += 1
                    else:
                        merged.append(right[j])
                        j += 1
                merged.extend(left[i:])
                merged.extend(right[j:])
                self.tree[v] = merged
        
        def query_less(self, l, r, x):
            # Query [l, r] (0-based, inclusive) for count of elements <x
            res = 0
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    res += bisect.bisect_left(self.tree[l], x)
                    l += 1
                if r % 2 == 0:
                    res += bisect.bisect_left(self.tree[r], x)
                    r -= 1
                l >>= 1
                r >>= 1
            return res

    st = SegmentTree(data)

    output = []
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        k = r + 1
        if k > N:
            output.append("0")
            continue
        left_idx = k - 1  # 0-based
        right_idx = N - 1
        if left_idx > right_idx:
            output.append("0")
        else:
            ans = st.query_less(left_idx, right_idx, l)
            output.append(str(ans))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()