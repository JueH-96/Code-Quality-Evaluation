import sys
import threading

def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    K = int(next(it))
    # Read A
    A = [0] * N
    mx = 0
    for i in range(N):
        x = int(next(it))
        A[i] = x
        if x > mx:
            mx = x

    # freq[x] = count of occurrences of x in A
    freq = [0] * (mx + 1)
    for x in A:
        freq[x] += 1

    # cnt[d] = number of elements in A divisible by d
    cnt = [0] * (mx + 1)
    # Localize for speed
    freq_loc = freq
    cnt_loc = cnt
    mx1 = mx
    rng = range
    for d in rng(1, mx1 + 1):
        s = 0
        # sum freq_loc[m] for m = d, 2d, 3d, ...
        for m in rng(d, mx1 + 1, d):
            s += freq_loc[m]
        cnt_loc[d] = s

    # f[v] = the maximum d dividing v with cnt[d] >= K
    f = [0] * (mx + 1)
    f_loc = f
    Kloc = K
    # Scan d from large to small so first assignment is the maximum
    for d in rng(mx1, 0, -1):
        if cnt_loc[d] >= Kloc:
            # assign f_loc[m] = d for multiples m of d not yet assigned
            for m in rng(d, mx1 + 1, d):
                if f_loc[m] == 0:
                    f_loc[m] = d

    # Output answers: for each A[i], print f[A[i]]
    out = sys.stdout.write
    for x in A:
        out(str(f_loc[x]))
        out('
')

if __name__ == "__main__":
    main()