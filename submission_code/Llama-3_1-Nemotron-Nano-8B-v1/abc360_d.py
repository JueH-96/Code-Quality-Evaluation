import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta=1):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    T = int(input[ptr])
    ptr +=1
    S = input[ptr]
    ptr +=1
    X = list(map(int, input[ptr:ptr+N]))
    ptr +=N

    scaled_X = [x * 10 for x in X]

    total = 0

    # Case 1: right i, left j, i < j, X_i < X_j, (X_j - X_i)/2 <= T+0.1
    # Process in original order, track right-moving ants in BIT
    right_X_case1 = []
    left_X_case1 = []
    lower_bound_case1 = []
    for i in range(N):
        if S[i] == '1':
            right_X_case1.append(scaled_X[i])
        else:
            left_X_case1.append(scaled_X[i])
            lower_bound_case1.append(scaled_X[i] - (20 * T + 2))
    all_values_case1 = right_X_case1 + left_X_case1 + lower_bound_case1
    sorted_values_case1 = sorted(list(set(all_values_case1)))
    compress_case1 = {v: i+1 for i, v in enumerate(sorted_values_case1)}  # 1-based
    bit_case1 = FenwickTree(len(sorted_values_case1))
    case1 = 0
    for i in range(N):
        if S[i] == '1':
            x = scaled_X[i]
            idx = compress_case1.get(x, 0)
            if idx != 0:
                bit_case1.update(idx)
        else:
            x_j = scaled_X[i]
            lb = x_j - (20 * T + 2)
            left_idx = bisect.bisect_left(sorted_values_case1, lb)
            right_idx = bisect.bisect_left(sorted_values_case1, x_j)
            if left_idx < len(sorted_values_case1) and right_idx > 0:
                l = left_idx + 1
                r = right_idx
                cnt = bit_case1.query(r - 1) - bit_case1.query(l - 1)
                case1 += cnt
    total += case1

    # Case 2: left i, right j, i < j, X_i > X_j, (X_i - X_j)/2 <= T+0.1
    left_X_case2 = []
    right_X_case2 = []
    upper_bound_case2 = []
    for i in range(N):
        if S[i] == '0':
            left_X_case2.append(scaled_X[i])
        else:
            right_X_case2.append(scaled_X[i])
            upper_bound_case2.append(scaled_X[i] + (20 * T + 2))
    all_values_case2 = left_X_case2 + right_X_case2 + upper_bound_case2
    sorted_values_case2 = sorted(list(set(all_values_case2)))
    compress_case2 = {v: i+1 for i, v in enumerate(sorted_values_case2)}
    bit_case2 = FenwickTree(len(sorted_values_case2))
    case2 = 0
    for i in range(N):
        if S[i] == '0':
            x = scaled_X[i]
            idx = compress_case2.get(x, 0)
            if idx != 0:
                bit_case2.update(idx)
        else:
            x_j = scaled_X[i]
            ub = x_j + (20 * T + 2)
            left_idx = bisect.bisect_left(sorted_values_case2, x_j + 1)
            right_idx = bisect.bisect_right(sorted_values_case2, ub)
            if left_idx < len(sorted_values_case2) and right_idx > 0:
                l = left_idx + 1
                r = right_idx
                cnt = bit_case2.query(r - 1) - bit_case2.query(l - 1)
                case2 += cnt
    total += case2
    print(total)

if __name__ == '__main__':
    main()