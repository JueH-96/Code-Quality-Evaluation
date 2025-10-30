import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        a = list(map(int, data[idx:idx+n]))
        idx += n

        # build prefix sums
        ps = [0] * (n+1)
        for i in range(n):
            ps[i+1] = ps[i] + a[i]

        # find all divisors k of n
        divisors = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.append(i)
                if i != n // i:
                    divisors.append(n // i)
            i += 1

        best = 0
        # for each possible k, compute max-min of segment sums of length k
        for k in divisors:
            # number of segments = n // k
            # compute sums ps[j*k] - ps[(j-1)*k] for j=1..n/k
            seg_min = 10**30
            seg_max = -1
            # slide by block
            for start in range(0, n, k):
                s = ps[start + k] - ps[start]
                if s < seg_min: seg_min = s
                if s > seg_max: seg_max = s
            diff = seg_max - seg_min
            if diff > best:
                best = diff

        out.append(str(best))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()