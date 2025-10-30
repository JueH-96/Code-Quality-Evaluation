import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, K = data[0], data[1]
    A = data[2:]

    max_val = max(A)

    # frequency of every value
    freq = [0] * (max_val + 1)
    for v in A:
        freq[v] += 1

    # cnt[d] = how many numbers are divisible by d
    cnt = [0] * (max_val + 1)
    for d in range(1, max_val + 1):
        s = 0
        for m in range(d, max_val + 1, d):
            s += freq[m]
        cnt[d] = s

    # ans[v] = largest divisor of v that appears in at least K numbers
    ans = [0] * (max_val + 1)
    for d in range(max_val, 0, -1):
        if cnt[d] >= K:                    # d is “good”
            for m in range(d, max_val + 1, d):
                if ans[m] == 0:            # first (thus largest) good divisor met
                    ans[m] = d

    out = '
'.join(str(ans[v]) for v in A)
    sys.stdout.write(out)

if __name__ == "__main__":
    main()