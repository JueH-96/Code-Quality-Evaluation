def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    idx = 1

    from math import sqrt

    out = []
    for _ in range(t):
        n = int(input_data[idx]); idx += 1
        arr = list(map(int, input_data[idx:idx+n]))
        idx += n

        # Compute prefix sums
        prefix = [0] * (n+1)
        for i in range(n):
            prefix[i+1] = prefix[i] + arr[i]

        # Function to get all divisors of n
        # We only need positive divisors
        divisors = []
        limit = int(sqrt(n))
        for d in range(1, limit+1):
            if n % d == 0:
                divisors.append(d)
                if d != n // d:
                    divisors.append(n // d)

        max_diff = 0

        # For each valid divisor k, partition in consecutive blocks of size k
        # Then compute min and max sums, track the difference
        for k in divisors:
            block_count = n // k
            if block_count == 1:
                # Means we have only one truck => difference = 0
                # No need to check sums
                continue
            # Compute sums quickly
            min_s = float('inf')
            max_s = -float('inf')
            start = 0
            for i in range(block_count):
                seg_sum = prefix[(i+1)*k] - prefix[i*k]
                if seg_sum < min_s:
                    min_s = seg_sum
                if seg_sum > max_s:
                    max_s = seg_sum
            diff = max_s - min_s
            if diff > max_diff:
                max_diff = diff

        out.append(str(max_diff))
    print("
".join(out))

# Let's call solve() to read from stdin and produce the result
# (Not strictly necessary on some competitive programming platforms,
#  but called here for completeness.)
if __name__ == "__main__":
    solve()