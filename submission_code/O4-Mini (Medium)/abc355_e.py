import sys

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N, L, R = map(int, data)
    ans = 0
    cur = L
    # Decompose [L, R] into dyadic intervals [l, r] where length = 2^i
    # and l is divisible by 2^i.
    while cur <= R:
        # find the largest i such that
        # 1) cur is aligned to 2^i, i.e. cur % (1<<i) == 0
        # 2) cur + (1<<i) - 1 <= R
        for i in range(N, -1, -1):
            length = 1 << i
            if cur % length == 0 and cur + length - 1 <= R:
                break
        j = cur >> i
        # ask the judge for the sum over [cur, cur+2^i-1] mod 100
        print(f"? {i} {j}", flush=True)
        line = sys.stdin.readline().strip()
        if not line:
            # no response; exit
            sys.exit(0)
        t = int(line)
        if t < 0:
            # invalid query or too many queries
            sys.exit(0)
        ans = (ans + t) % 100
        cur += length

    # output the final answer
    print(f"! {ans}", flush=True)

main()