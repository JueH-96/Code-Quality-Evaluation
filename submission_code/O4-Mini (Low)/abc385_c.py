import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    # If there's only one building, answer is 1
    if N == 1:
        print(1)
        return

    ans = 1  # At least one building can always be chosen

    # For each possible interval d between chosen buildings
    for d in range(1, N):
        # We consider sequences starting at offsets 0,1,...,d-1
        for start in range(d):
            prev_h = None
            cnt = 0
            # Walk through positions start, start+d, start+2d, ...
            for pos in range(start, N, d):
                h = H[pos]
                if h == prev_h:
                    cnt += 1
                else:
                    prev_h = h
                    cnt = 1
                if cnt > ans:
                    ans = cnt

    print(ans)

if __name__ == "__main__":
    main()