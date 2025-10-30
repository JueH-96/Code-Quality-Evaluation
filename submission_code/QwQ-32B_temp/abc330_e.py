import sys

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [float('inf')] * (2 * self.n)
    
    def update(self, pos, value):
        pos += self.n
        self.tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = min(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
            pos >>= 1
    
    def query(self):
        return self.tree[1]

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    count = [0] * N
    present = [False] * N

    for val in A:
        if val < N:
            count[val] += 1

    for x in range(N):
        present[x] = (count[x] > 0)

    st = SegmentTree(N)
    for x in range(N):
        if not present[x]:
            val = x
        else:
            val = float('inf')
        st.update(x, val)

    for _ in range(Q):
        i = int(input[idx]) - 1  # convert to 0-based
        idx += 1
        x_new = int(input[idx])
        idx += 1

        old_val = A[i]
        new_val = x_new

        # Process old_val
        if old_val < N:
            count[old_val] -= 1
            if count[old_val] == 0:
                present[old_val] = False
            else:
                present[old_val] = True
            # Update the segment tree
            if present[old_val]:
                val = float('inf')
            else:
                val = old_val
            st.update(old_val, val)

        # Process new_val
        if new_val < N:
            prev_count = count[new_val]
            count[new_val] += 1
            # present is now True
            val = float('inf')
            st.update(new_val, val)

        A[i] = new_val

        first_missing = st.query()
        if first_missing == float('inf'):
            print(N)
        else:
            print(first_missing)

if __name__ == "__main__":
    main()