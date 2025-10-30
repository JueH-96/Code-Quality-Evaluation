import sys

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [1] * (2 * self.size)  # Initialize all to 1

        # Fill the leaves with data
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # Fill remaining leaves with 1 (they are beyond the data)
        for i in range(self.n, self.size):
            self.tree[self.size + i] = 1

        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] & self.tree[2 * i + 1]

    def update_point(self, pos, new_val):
        pos += self.size
        self.tree[pos] = new_val
        pos >>= 1
        while pos >= 1:
            new_val = self.tree[2 * pos] & self.tree[2 * pos + 1]
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1

    def query_range(self, l, r):
        res = 1
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res &= self.tree[l]
                l += 1
            if r % 2 == 0:
                res &= self.tree[r]
                r -= 1
            l >>= 1
            r >>= 1
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr]); ptr +=1
    Q = int(input[ptr]); ptr +=1
    S = input[ptr]; ptr +=1

    # Compute T array
    T = []
    for i in range(N-1):
        if S[i] != S[i+1]:
            T.append(1)
        else:
            T.append(0)

    # Initialize segment tree if T is not empty
    if T:
        st = SegmentTree(T)
    else:
        st = None  # For N=1, no T array exists

    for _ in range(Q):
        typ = int(input[ptr]); ptr +=1
        L = int(input[ptr]); ptr +=1
        R = int(input[ptr]); ptr +=1

        if typ == 1:
            a = L - 1  # 0-based start of flip in S
            b = R - 1  # 0-based end of flip in S

            # Toggle T[a-1] if a > 0
            if a > 0:
                pos = a - 1
                current = T[pos]
                new_val = 1 - current
                T[pos] = new_val
                st.update_point(pos, new_val)
            # Toggle T[b] if b < N-1 (since T has indices up to N-2)
            if b < (N - 1):
                pos = b
                current = T[pos]
                new_val = 1 - current
                T[pos] = new_val
                st.update_point(pos, new_val)
        else:
            # Type 2 query
            a = L - 1  # 0-based start of substring in S
            b = R - 1  # 0-based end of substring in S
            start = a
            end = b - 1

            if start > end:
                print("Yes")
            else:
                if not T:  # N=1 case
                    print("Yes")
                else:
                    res = st.query_range(start, end)
                    print("Yes" if res == 1 else "No")

if __name__ == "__main__":
    main()