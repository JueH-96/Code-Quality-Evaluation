import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    R = [int(next(it)) for _ in range(n)]

    out = []
    seq = [0] * n

    def dfs(idx, cur_sum_mod):
        if idx == n:
            if cur_sum_mod % k == 0:
                # append formatted sequence
                out.append(" ".join(map(str, seq)))
            return
        # try all values from 1 to R[idx]
        for v in range(1, R[idx] + 1):
            seq[idx] = v
            dfs(idx + 1, (cur_sum_mod + v) % k)

    dfs(0, 0)

    # Print all results, one per line
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()