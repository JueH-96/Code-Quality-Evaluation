import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    # We will for each test compute pref_max and suff_min and look for a k with
    # pref_max[k-1]==k-1 and suff_min[k+1]==k+1. If the permutation is already
    # sorted, answer=0; else if such k exists, answer=1; otherwise answer=2.
    for _ in range(T):
        n = int(next(it))
        P = [0]*(n+1)
        sorted_already = True
        for i in range(1, n+1):
            v = int(next(it))
            P[i] = v
            if v != i:
                sorted_already = False
        if sorted_already:
            out.append('0')
            continue
        # build prefix max
        pref_max = [0]*(n+1)
        m = 0
        for i in range(1, n+1):
            if P[i] > m:
                m = P[i]
            pref_max[i] = m
        # build suffix min
        suff_min = [0]*(n+2)
        m = n+1
        for i in range(n, 0, -1):
            if P[i] < m:
                m = P[i]
            suff_min[i] = m
        suff_min[n+1] = n+1
        # try to find a k with pref_max[k-1]==k-1 and suff_min[k+1]==k+1
        found = False
        # we index k from 1 to n
        for k in range(1, n+1):
            if pref_max[k-1] == k-1 and suff_min[k+1] == k+1:
                found = True
                break
        if found:
            out.append('1')
        else:
            out.append('2')
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()