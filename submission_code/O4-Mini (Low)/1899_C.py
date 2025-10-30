import sys
import threading

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    idx = 1
    out = []
    for _ in range(t):
        n = int(data[idx]); idx += 1
        # We'll do a variant of Kadane's algorithm that only extends
        # when parity alternates, otherwise we restart.
        dp_prev = None  # best sum ending at previous position
        prev_parity = None
        best = -10**18
        for i in range(n):
            a = int(data[idx]); idx += 1
            p = a & 1
            if dp_prev is None:
                # first element in this test
                dp = a
            else:
                # if parity alternates, we can extend dp_prev if positive
                if p != prev_parity:
                    # only extend if it helps (dp_prev positive)
                    ext = dp_prev if dp_prev > 0 else 0
                    dp = a + ext
                else:
                    # same parity: must restart
                    dp = a
            # track best
            if dp > best:
                best = dp
            dp_prev = dp
            prev_parity = p
        out.append(str(best))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()