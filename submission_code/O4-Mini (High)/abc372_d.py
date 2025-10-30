import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    H = list(map(int, data[1:]))

    # Compute for each i the index of the previous greater element (to the left)
    stack = []
    L = [-1] * n
    for i in range(n):
        h = H[i]
        while stack and H[stack[-1]] < h:
            stack.pop()
        L[i] = stack[-1] if stack else -1
        stack.append(i)

    # Use a difference array to accumulate contributions
    d = [0] * n
    # For each building j = i (0-based) from 1..n-1, it contributes +1
    # to all c[k] for k in [start..i-1], where start = max(0, L[i])
    for i in range(1, n):
        start = L[i] if L[i] >= 0 else 0
        d[start] += 1
        d[i] -= 1

    # Prefix sum to get final counts c[i]
    c = [0] * n
    running = 0
    for i in range(n):
        running += d[i]
        c[i] = running

    # Output
    sys.stdout.write(" ".join(map(str, c)))

if __name__ == "__main__":
    main()