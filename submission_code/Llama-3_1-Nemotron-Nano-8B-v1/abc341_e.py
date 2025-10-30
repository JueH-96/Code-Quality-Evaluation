import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, delta):
        idx += 1  # Convert to 1-based index
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        idx += 1  # Convert to 1-based index
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    current_B = []
    for i in range(N-1):
        current_B.append(1 if S[i] == S[i+1] else 0)
    
    bit = FenwickTree(len(current_B))
    for i in range(len(current_B)):
        bit.update(i, current_B[i])

    for _ in range(Q):
        query_type = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if query_type == 1:
            L0 = L - 1
            R0 = R - 1
            pos_list = []
            if L0 > 0:
                pos_list.append(L0 - 1)
            if R0 < N - 1:
                pos_list.append(R0)
            for pos in pos_list:
                delta = 1 - 2 * current_B[pos]
                bit.update(pos, delta)
                current_B[pos] ^= 1
        else:
            if L == R:
                print("Yes")
            else:
                a = L - 1
                b = R - 2
                sum_val = bit.query(b)
                if a > 0:
                    sum_val -= bit.query(a - 1)
                print("Yes" if sum_val == 0 else "No")

if __name__ == '__main__':
    main()