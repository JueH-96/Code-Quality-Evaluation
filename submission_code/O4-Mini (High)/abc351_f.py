import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    # read the sequence
    A = list(map(int, data[1:]))

    # coordinate compression
    vals = sorted(set(A))
    m = len(vals)
    comp = {v: i+1 for i, v in enumerate(vals)}  # 1-based indices for BIT

    # Fenwick trees for counts and prefix‐sums
    cnt_tree = [0] * (m+1)
    sum_tree = [0] * (m+1)

    ans = 0
    # for speed, all these names are local
    cnt = cnt_tree
    sm = sum_tree
    for x in A:
        idx = comp[x]

        # query how many previous values < x, and their sum
        i = idx - 1
        c = 0
        s = 0
        while i > 0:
            c += cnt[i]
            s += sm[i]
            i -= i & -i

        # each of those contributes (x - Ai)
        ans += x * c - s

        # update BITs at position idx
        i = idx
        while i <= m:
            cnt[i] += 1
            sm[i] += x
            i += i & -i

    # output the result
    sys.stdout.write(str(ans))

main()