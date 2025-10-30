# YOUR CODE HERE
import sys
import threading
def main():
    import bisect
    import sys
    input = sys.stdin.readline

    sys.setrecursionlimit(1<<25)

    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))

    N = len(A_list)
    A = A_list

    # Coordinate compression
    A_values = sorted(set(A))
    value_to_compressed = {v:i+1 for i,v in enumerate(A_values)}  # compressed indices start from 1
    compressed_A = [value_to_compressed[a] for a in A]
    max_compressed = len(A_values) + 2  # add extra space

    class BIT:
        def __init__(self, size):
            self.N = size + 2
            self.tree = [0] * (self.N)

        def update(self, idx, value):
            while idx < self.N:
                self.tree[idx] += value
                idx += idx & -idx

        def query(self, idx):
            res = 0
            while idx > 0:
                res += self.tree[idx]
                idx -= idx & -idx
            return res

    N = len(A)

    BIT_count = BIT(max_compressed)
    BIT_sum = BIT(max_compressed)

    total_answer = 0
    sum_total = 0
    cnt_total = 0

    for i in range(N-1, -1, -1):
        a_i = A[i]
        a_i_compressed = compressed_A[i]
        sum_less_equal = BIT_sum.query(a_i_compressed)
        cnt_less_equal = BIT_count.query(a_i_compressed)

        sum_greater = sum_total - sum_less_equal
        cnt_greater = cnt_total - cnt_less_equal

        contrib_i = sum_greater - a_i * cnt_greater
        total_answer += contrib_i

        BIT_sum.update(a_i_compressed, a_i)
        BIT_count.update(a_i_compressed, 1)

        sum_total += a_i
        cnt_total += 1
    print(total_answer)

threading.Thread(target=main,).start()