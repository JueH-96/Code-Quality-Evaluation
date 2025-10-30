import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0

    N = int(data[idx])
    idx += 1

    A = list(map(int, data[idx:idx + N]))
    idx += N

    m = (N - 1) // 2
    s = []
    e = []
    for k in range(m):
        s.append(A[2 * k + 1])
        e.append(A[2 * k + 2])

    # Build prefix sums
    prefix_s = [0] * (m + 1)
    prefix_e = [0] * (m + 1)
    for i in range(m):
        prefix_s[i + 1] = prefix_s[i] + s[i]
        prefix_e[i + 1] = prefix_e[i] + e[i]

    Q = int(data[idx])
    idx += 1

    output = []
    for _ in range(Q):
        l = int(data[idx])
        r = int(data[idx + 1])
        idx += 2

        if l >= r:
            output.append('0')
            continue

        # Compute left and right
        left = bisect.bisect_right(e, l)
        right = bisect.bisect_left(s, r)

        if left >= right:
            output.append('0')
            continue

        # Compute sum_min_e
        x_max = bisect.bisect_right(e, r) - 1
        x_max_in_range = min(x_max, right - 1)
        if x_max_in_range < left:
            part_e_total = (right - left) * r
        else:
            part_e_total = prefix_e[x_max_in_range + 1] - prefix_e[left]
            part_e_r = ((right - 1) - x_max_in_range) * r
            part_e_total += part_e_r

        # Compute sum_max_s
        y_min = bisect.bisect_left(s, l)
        y_min_in_range = max(y_min, left)
        if y_min_in_range >= right:
            sum_max_s = (right - left) * l
        else:
            sum_max_s = (y_min_in_range - left) * l + (prefix_s[right] - prefix_s[y_min_in_range])

        total = part_e_total - sum_max_s
        ans = max(0, total)
        output.append(str(ans))

    print('
'.join(output))

if __name__ == '__main__':
    main()