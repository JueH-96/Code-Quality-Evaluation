import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    H = list(map(int, data[1:]))

    # diff array for range-add, 1-indexed up to N
    diff = [0] * (N + 2)

    stack = []  # will store indices (1-based) of a monotonic decreasing stack
    # find for each j the previous greater index p, then add +1 to c[i] for i in [p..j-1]
    for j0 in range(N):
        h_j = H[j0]
        j = j0 + 1  # 1-based
        while stack and H[stack[-1]-1] <= h_j:
            stack.pop()
        p = stack[-1] if stack else 0
        # add +1 on [max(1,p) .. j-1]
        l = p if p >= 1 else 1
        r = j - 1
        if l <= r:
            diff[l] += 1
            diff[r+1] -= 1
        stack.append(j)

    # build c by prefix-summing diff
    c = [0] * (N + 1)
    running = 0
    for i in range(1, N+1):
        running += diff[i]
        c[i] = running

    # output c[1..N]
    out = " ".join(str(c[i]) for i in range(1, N+1))
    sys.stdout.write(out)

if __name__ == "__main__":
    main()