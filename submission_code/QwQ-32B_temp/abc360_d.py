import bisect

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    T = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    X = list(map(int, input[ptr:ptr+N]))
    ptr += N

    R = []
    L = []
    for s, x in zip(S, X):
        if s == '1':
            R.append(x)
        else:
            L.append(x)
    R.sort()
    L.sort()
    total = 0
    for x_r in R:
        upper = x_r + 2 * (T + 0.1)
        cnt1 = bisect.bisect_right(L, x_r)
        cnt2 = bisect.bisect_right(L, upper)
        total += (cnt2 - cnt1)
    print(total)

if __name__ == "__main__":
    main()