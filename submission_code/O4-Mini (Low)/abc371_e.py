import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # last_pos[x] will store the last position (1-based) where x appeared
    # initialize to 0 meaning "not appeared yet"
    last_pos = [0] * (N + 1)

    S = 0   # S(r) = sum_{l=1..r} f(l, r)
    ans = 0 # final answer = sum_{r=1..N} S(r)

    # iterate r from 1 to N
    for r, x in enumerate(A, start=1):
        prev = last_pos[x]
        # number of l in [1..r] for which x is new in subarray [l..r]
        delta = r - prev

        S += delta
        ans += S

        last_pos[x] = r

    print(ans)

if __name__ == "__main__":
    main()