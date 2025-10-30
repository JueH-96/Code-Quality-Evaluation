def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    max_val = max(A)

    # Frequency array for values up to max_val
    freq = [0] * (max_val + 1)
    for v in A:
        freq[v] += 1

    # Prefix sum of values (v * freq[v])
    prefix_sum_val = [0] * (max_val + 1)
    for v in range(1, max_val + 1):
        prefix_sum_val[v] = prefix_sum_val[v - 1] + v * freq[v]

    total_sum = prefix_sum_val[max_val]

    # For each A[i], sum of elements greater than A[i] = total_sum - prefix_sum_val[A[i]]
    result = [str(total_sum - prefix_sum_val[val]) for val in A]

    print(" ".join(result))

# Let's call solve() to run the solution
# solve()