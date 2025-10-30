import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    W = list(map(int, input[ptr:ptr + N]))
    ptr += N
    L = []
    R = []
    for _ in range(N):
        li = int(input[ptr])
        ri = int(input[ptr + 1])
        L.append(li)
        R.append(ri)
        ptr += 2

    # Prepare sorted_r and prefix_min_r
    sorted_r = sorted([(R[i], W[i]) for i in range(N)], key=lambda x: x[0])
    sorted_R = [x[0] for x in sorted_r]
    sorted_weights_r = [x[1] for x in sorted_r]
    prefix_min_r = []
    if N > 0:
        prefix_min_r = [0] * N
        prefix_min_r[0] = sorted_weights_r[0]
        for i in range(1, N):
            prefix_min_r[i] = min(prefix_min_r[i - 1], sorted_weights_r[i])

    # Prepare sorted_L and suffix_min_l
    sorted_L = sorted([(L[i], W[i]) for i in range(N)], key=lambda x: x[0])
    sorted_L_list = [x[0] for x in sorted_L]
    sorted_L_weights = [x[1] for x in sorted_L]
    suffix_min_l = [0] * N
    if N > 0:
        suffix_min_l[N - 1] = sorted_L_weights[N - 1]
        for i in range(N - 2, -1, -1):
            suffix_min_l[i] = min(sorted_L_weights[i], suffix_min_l[i + 1])

    Q = int(input[ptr])
    ptr += 1
    output = []
    INF = float('inf')

    for _ in range(Q):
        s = int(input[ptr]) - 1
        t = int(input[ptr + 1]) - 1
        ptr += 2

        L_s = L[s]
        R_s = R[s]
        L_t = L[t]
        R_t = R[t]

        # Check for direct edge
        if R_s < L_t or L_s > R_t:
            res = W[s] + W[t]
        else:
            X = min(L_s, L_t)
            Y = max(R_s, R_t)

            # Compute best_left
            pos = bisect.bisect_left(sorted_R, X)
            best_left = INF
            if pos > 0:
                best_left = prefix_min_r[pos - 1]

            # Compute best_right
            pos2 = bisect.bisect_right(sorted_L_list, Y)
            best_right = INF
            if pos2 < N:
                best_right = suffix_min_l[pos2]

            min_candidate = min(best_left, best_right)
            if min_candidate == INF:
                res = -1
            else:
                res = W[s] + W[t] + min_candidate

        output.append(str(res))

    print('
'.join(output))

if __name__ == '__main__':
    main()