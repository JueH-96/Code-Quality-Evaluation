import sys
import threading

def main():
    import sys
    from bisect import bisect_left, bisect_right

    data = sys.stdin.read().split()
    N = int(data[0])
    Q = int(data[1])
    S = data[2]
    # Build prefix sums for '1' and '2'
    onesum = [0] * (N + 1)
    twosum = [0] * (N + 1)
    slash_pos = []
    for i, ch in enumerate(S, start=1):
        onesum[i] = onesum[i-1] + (1 if ch == '1' else 0)
        twosum[i] = twosum[i-1] + (1 if ch == '2' else 0)
        if ch == '/':
            slash_pos.append(i)

    out = []
    idx = 3
    for _ in range(Q):
        L = int(data[idx]); R = int(data[idx+1])
        idx += 2
        # find slash positions in [L,R]
        li = bisect_left(slash_pos, L)
        ri = bisect_right(slash_pos, R) - 1
        if li > ri:
            out.append('0')
            continue

        # helper to test A[j] <= B[j]
        # where j is index in slash_pos
        def cond(j):
            p = slash_pos[j]
            # ones before p but not before L
            a = onesum[p-1] - onesum[L-1]
            # twos after p but not after R
            b = twosum[R] - twosum[p]
            return a <= b

        # binary search for last j in [li, ri] with cond(j) True
        low, high = li, ri
        last_true = li - 1
        while low <= high:
            mid = (low + high) // 2
            if cond(mid):
                last_true = mid
                low = mid + 1
            else:
                high = mid - 1

        best = 0
        # candidate 1: j = last_true, k = A
        j1 = last_true
        if li <= j1 <= ri:
            p = slash_pos[j1]
            a = onesum[p-1] - onesum[L-1]
            # k = a
            best = max(best, 2 * a + 1)
        # candidate 2: j = last_true + 1, k = B
        j2 = last_true + 1
        if li <= j2 <= ri:
            p = slash_pos[j2]
            b = twosum[R] - twosum[p]
            best = max(best, 2 * b + 1)

        out.append(str(best))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()