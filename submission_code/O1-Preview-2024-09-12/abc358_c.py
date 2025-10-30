import sys
import itertools

def main():
    N, M = map(int, sys.stdin.readline().split())
    stands = []
    for _ in range(N):
        s = sys.stdin.readline().strip()
        bitmask = 0
        for i, c in enumerate(s):
            if c == 'o':
                bitmask |= 1 << i
        stands.append(bitmask)
    full_mask = (1 << M) -1

    for k in range(1, N+1):
        found = False
        for combo in itertools.combinations(stands, k):
            union = 0
            for bm in combo:
                union |= bm
            if union == full_mask:
                print(k)
                return
    # If we reach here, we need all N stands
    print(N)

main()