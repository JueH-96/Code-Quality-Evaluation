import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    pos = 0
    t = data[pos]
    pos += 1
    out_lines = []

    for _ in range(t):
        n = data[pos]
        pos += 1
        a = data[pos:pos + n]
        pos += n

        # Kadane–style DP with parity constraint
        best = cur = a[0]
        for i in range(1, n):
            if (a[i] & 1) ^ (a[i - 1] & 1):           # different parity -> may extend
                cur = max(a[i], cur + a[i])
            else:                                     # same parity -> must restart
                cur = a[i]
            if cur > best:
                best = cur

        out_lines.append(str(best))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()