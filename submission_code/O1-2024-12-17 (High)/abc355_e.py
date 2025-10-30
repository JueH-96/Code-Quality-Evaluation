def main():
    import sys

    # Interactive "ask" function: prints a query "? i j", flushes,
    # then reads the integer T. If T == -1, we terminate immediately.
    def ask(i, j):
        print(f"? {i} {j}", flush=True)
        T_line = sys.stdin.readline().strip()
        if not T_line:
            # In a real interactive environment this shouldn't happen,
            # but just in case we handle an unexpected EOF or empty response.
            sys.exit(0)
        T = int(T_line)
        if T == -1:
            sys.exit(0)  # we must terminate if T == -1
        return T

    # Returns sum(A[0] + A[1] + ... + A[K]) mod 100
    # by decomposing [0..K] into power-of-two-aligned blocks.
    def get_partial_sum(K):
        if K < 0:
            return 0
        total_mod = 0
        x = K + 1  # We'll decrement x by power-of-two blocks
        while x > 0:
            # lowest power-of-two dividing x
            lowest_bit = x & (-x)          # e.g., if x=12(1100), lowest_bit=4(0100)
            i = lowest_bit.bit_length() - 1  # position of that bit
            block_size = 1 << i             # 2^i
            start = x - block_size
            j = start >> i  # since start = 2^i * j
            # Ask the query for sum(A[start]..A[x-1]) mod 100
            val = ask(i, j)
            total_mod = (total_mod + val) % 100
            x -= block_size
        return total_mod

    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data)

    # Compute sum(A[L]..A[R]) mod 100 = (sum(A[0]..A[R]) - sum(A[0]..A[L-1])) mod 100
    sum_left_minus_one = get_partial_sum(L - 1)
    sum_right = get_partial_sum(R)
    answer = (sum_right - sum_left_minus_one) % 100

    # Print the final answer in the required format and flush
    print(f"! {answer}", flush=True)

# Do not forget to call main().
main()