class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.size:
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
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Compute mod_p
    mod_p = [0]
    current_sum = 0
    for a in A:
        current_sum += a
        mod_p.append(current_sum % M)

    # Compute sum_a
    sum_a = [0] * (len(mod_p) + 1)
    sum_a[0] = 0
    current_sum = 0
    for i in range(len(mod_p)):
        sum_a[i + 1] = current_sum
        current_sum += mod_p[i]

    # Initialize BIT
    bit = BIT(M)
    # Add mod_p[0] to BIT
    bit.update(mod_p[0] + 1, 1)

    total = 0

    for r in range(1, N + 1):
        x = mod_p[r]
        current_count = r
        sum_part = x * current_count - sum_a[r]
        q = bit.query(x + 1)
        cnt_gt = current_count - q
        contribution = sum_part + M * cnt_gt
        total += contribution
        # Add x to BIT
        bit.update(x + 1, 1)

    print(total)

if __name__ == '__main__':
    main()