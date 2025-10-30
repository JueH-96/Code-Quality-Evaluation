import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    h = list(map(int, data[1:]))

    # difference array for range-add / point-query
    diff = [0] * (n + 3)          # indices 1 … n  (n+2 is safe for r+1)

    stack = []                    # monotonically decreasing (stores indices, 1-based)
    for j in range(1, n + 1):     # j : current building index (1-based)
        cur_h = h[j - 1]

        # pop every lower building – the remainder's top is the previous taller one
        while stack and h[stack[-1] - 1] < cur_h:
            stack.pop()

        prev_taller = stack[-1] if stack else 0   # p(j)

        # j contributes to every i with prev_taller ≤ i < j
        l = max(1, prev_taller)   # i starts from 1
        r = j - 1
        if l <= r:
            diff[l] += 1
            diff[r + 1] -= 1

        stack.append(j)

    # prefix sum of diff gives required counts
    res = []
    running = 0
    for i in range(1, n + 1):
        running += diff[i]
        res.append(str(running))

    sys.stdout.write(" ".join(res))


if __name__ == "__main__":
    main()