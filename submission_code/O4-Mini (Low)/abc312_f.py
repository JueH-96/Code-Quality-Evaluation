import sys
import threading

def main():
    import sys
    data = sys.stdin
    N, M = map(int, data.readline().split())
    A = []  # pull-tab cans (type 0)
    B = []  # regular cans (type 1)
    O = []  # can openers (type 2)
    for _ in range(N):
        t, x = map(int, data.readline().split())
        if t == 0:
            A.append(x)
        elif t == 1:
            B.append(x)
        else:
            O.append(x)
    # sort descending
    A.sort(reverse=True)
    B.sort(reverse=True)
    O.sort(reverse=True)
    # build prefix sums
    # sumA[i] = sum of top i pull-tabs
    sumA = [0] * (len(A) + 1)
    for i, v in enumerate(A, start=1):
        sumA[i] = sumA[i-1] + v
    # sumB[i] = sum of top i regular cans
    sumB = [0] * (len(B) + 1)
    for i, v in enumerate(B, start=1):
        sumB[i] = sumB[i-1] + v
    # sumO[i] = total capacity of top i openers
    sumO = [0] * (len(O) + 1)
    for i, v in enumerate(O, start=1):
        sumO[i] = sumO[i-1] + v
    import bisect
    best = 0
    maxB = min(len(B), M)
    # iterate how many regular cans we open
    for k1 in range(0, maxB + 1):
        # need openers capacity >= k1
        # find smallest k2 such that sumO[k2] >= k1
        # sumO is non-decreasing
        k2 = bisect.bisect_left(sumO, k1)
        if k2 > len(O):
            continue
        # total items used so far: k1 regular + k2 openers
        rem = M - k1 - k2
        if rem < 0:
            continue
        # rem slots must be filled by pull-tabs
        if rem > len(A):
            continue
        # compute total happiness
        val = sumB[k1] + sumA[rem]
        if val > best:
            best = val
    print(best)

if __name__ == "__main__":
    main()