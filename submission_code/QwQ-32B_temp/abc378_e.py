class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N

    # Compute prefix sums S
    S = [0] * (N + 1)
    for i in range(1, N + 1):
        S[i] = S[i - 1] + A[i - 1]

    # Compute q and s arrays
    q = [0] * (N + 1)
    s = [0] * (N + 1)
    for i in range(N + 1):
        q[i] = S[i] // M
        s[i] = S[i] % M

    # Compute prefix_q array
    prefix_q = [0] * (N + 1)
    prefix_q[0] = q[0]
    for i in range(1, N + 1):
        prefix_q[i] = prefix_q[i - 1] + q[i]

    # Compute sum1 and sum2 for total_subarrays_sum
    sum1 = 0
    for r in range(1, N + 1):
        sum1 += S[r] * r

    sum2 = 0
    for k in range(0, N):
        sum2 += S[k] * (N - k)

    total_subarrays = sum1 - sum2

    # Fenwick Tree part
    ft_size = M
    fen = FenwickTree(ft_size)
    # Initial s[0] is 0 mod M
    fen.update(s[0] + 1, 1)  # s[0] is 0 → index 1
    current_count = 1
    total_floor = 0

    for r in range(1, N + 1):
        current_s = s[r]
        total_so_far = current_count
        # Compute count_le = query(s_r +1)
        count_le = fen.query(current_s + 1)
        count_less = total_so_far - count_le

        # Compute terms
        q_r = q[r]
        term1 = q_r * r - prefix_q[r - 1]
        term2 = -count_less
        total_floor += (term1 + term2)

        # Update Fenwick Tree with current_s
        fen.update(current_s + 1, 1)
        current_count += 1

    ans = total_subarrays - M * total_floor
    print(ans)

if __name__ == "__main__":
    main()