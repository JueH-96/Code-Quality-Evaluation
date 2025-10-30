import sys
import threading

def main():
    import sys
    from bisect import bisect_left

    data = sys.stdin.read().split()
    t = int(data[0])
    ans = []
    idx = 1
    for _ in range(t):
        n = int(data[idx]); idx += 1
        k = int(data[idx]); idx += 1
        A = list(map(int, data[idx:idx+n]))
        idx += n
        B = list(map(int, data[idx:idx+n]))
        idx += n

        # build positions of each value in A
        pos = {}
        for i, v in enumerate(A):
            if v not in pos:
                pos[v] = []
            pos[v].append(i)

        ok = True
        for i, v in enumerate(B):
            if v not in pos:
                ok = False
                break
            lst = pos[v]
            # window [i-k, i+k]
            lo = i - k
            hi = i + k
            # binary search for first position >= lo
            j = bisect_left(lst, lo)
            # check if this position <= hi
            if j == len(lst) or lst[j] > hi:
                ok = False
                break

        ans.append("Yes" if ok else "No")

    sys.stdout.write("
".join(ans))

if __name__ == "__main__":
    main()