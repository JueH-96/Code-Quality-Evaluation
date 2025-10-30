import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, idx, val):
        while idx <= self.n:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Initialize Fenwick Trees
    count_bit = FenwickTree(M)
    sum_bit = FenwickTree(M)

    # Initial prefix is 0
    initial_v = 0
    count_bit.update(initial_v + 1, 1)
    sum_bit.update(initial_v + 1, initial_v)

    answer = 0
    prev_prefix = 0

    for a in A:
        new_prefix = (prev_prefix + a) % M
        curr = new_prefix

        # Query total count and sum
        total_count = count_bit.query(M)
        sum_x_total = sum_bit.query(M)

        # Compute count_up_to curr + 1
        count_up_to = count_bit.query(curr + 1)
        number_of_x_greater = total_count - count_up_to

        contribution = curr * total_count - sum_x_total + number_of_x_greater * M
        answer += contribution

        # Update the bits
        count_bit.update(curr + 1, 1)
        sum_bit.update(curr + 1, curr)

        # Update prev_prefix
        prev_prefix = new_prefix

    print(answer)

if __name__ == '__main__':
    main()