# YOUR CODE HERE
def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    X = list(map(int, input_data[1:N+1]))
    P = list(map(int, input_data[N+1:2*N+1]))

    # Build prefix sums
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + P[i]

    Q = int(input_data[2*N+1])
    idx = 2*N + 2
    out = []
    for _ in range(Q):
        L = int(input_data[idx]); R = int(input_data[idx+1])
        idx += 2
        left = bisect.bisect_left(X, L)
        right = bisect.bisect_right(X, R) - 1
        if left <= right:
            out.append(str(prefix_sum[right+1] - prefix_sum[left]))
        else:
            out.append('0')

    print('
'.join(out))

# Don't forget to call main()
main()