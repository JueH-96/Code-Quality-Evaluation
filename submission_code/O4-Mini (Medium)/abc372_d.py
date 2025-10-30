import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    H = [0] * (n + 1)
    for i in range(1, n+1):
        H[i] = int(next(it))
    # Compute L[j] = index of previous greater element to the left of j (0 if none)
    L = [0] * (n + 1)
    stack = []
    for j in range(1, n+1):
        # pop until we find a greater height or stack empty
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        L[j] = stack[-1] if stack else 0
        stack.append(j)
    # We'll do range-add on c[i] for i in [l..r] for each j, then prefix-sum
    # diff array for 1..n, of size n+2 to avoid bounds checks
    diff = [0] * (n + 2)
    for j in range(2, n+1):
        l = L[j] if L[j] >= 1 else 1
        r = j - 1
        if l <= r:
            diff[l] += 1
            diff[r+1] -= 1
    # build c[i] by prefix-sum of diff
    res = [0] * (n + 1)
    cur = 0
    out = []
    for i in range(1, n+1):
        cur += diff[i]
        out.append(str(cur))
    sys.stdout.write(" ".join(out))

if __name__ == "__main__":
    main()