import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    t = int(next(it))
    out = []
    for _ in range(t):
        n = int(next(it))
        dp_prev = 0
        prev_parity = None
        ans = -10**18
        for _ in range(n):
            a = int(next(it))
            parity = a & 1
            if prev_parity is not None and parity != prev_parity:
                # can extend, but only if it improves over starting fresh
                dp_curr = a + dp_prev
                if dp_curr < a:
                    dp_curr = a
            else:
                dp_curr = a
            # update answer
            if dp_curr > ans:
                ans = dp_curr
            # carry forward
            dp_prev = dp_curr
            prev_parity = parity
        out.append(str(ans))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()