import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    # Store for each die: its face count dict and K_i
    counts = []
    ks = []
    for _ in range(n):
        k = int(next(it))
        ks.append(k)
        cnt = {}
        for _ in range(k):
            a = int(next(it))
            cnt[a] = cnt.get(a, 0) + 1
        counts.append(cnt)

    max_prob = 0.0
    # For each pair of dice
    for i in range(n):
        ci = counts[i]
        ki = ks[i]
        # Pre-select to iterate smaller dict for speed
        for j in range(i+1, n):
            cj = counts[j]
            kj = ks[j]
            # choose smaller dictionary to iterate
            if len(ci) <= len(cj):
                small, big = ci, cj
            else:
                small, big = cj, ci
            common = 0
            for face, cnt_small in small.items():
                cnt_big = big.get(face)
                if cnt_big:
                    common += cnt_small * cnt_big
            prob = common / (ki * kj)
            if prob > max_prob:
                max_prob = prob

    # Print with sufficient precision
    print("{:.12f}".format(max_prob))

if __name__ == "__main__":
    main()