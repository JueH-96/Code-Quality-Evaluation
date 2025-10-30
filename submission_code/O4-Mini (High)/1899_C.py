import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out = []
    for _ in range(t):
        n = int(next(it))
        # initialize with the first element
        x = int(next(it))
        cur = x
        ans = x
        prev = x
        # process the rest
        for _ in range(1, n):
            x = int(next(it))
            # if parity alternates, we can choose to extend or restart
            if (x & 1) != (prev & 1):
                # cur = max(x, cur + x)  equivalent to x + max(cur, 0)
                if cur > 0:
                    cur += x
                else:
                    cur = x
            else:
                # same parity, must restart
                cur = x
            # track the maximum sum seen
            if cur > ans:
                ans = cur
            prev = x
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()