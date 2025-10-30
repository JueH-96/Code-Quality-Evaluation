import sys
import threading

def main():
    data = sys.stdin.readline().split()
    if not data:
        return
    N = int(data[0])
    S = data[1].rstrip()

    # coverage[i] will count how many matched-intervals cover position i
    # we use a difference array of length N+1
    coverage = [0] * (N + 1)
    stack = []

    # Find matched parentheses and mark their intervals in coverage
    for i, ch in enumerate(S):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                l = stack.pop()
                coverage[l] += 1
                # end is i, so decrement at i+1
                coverage[i+1] -= 1

    # Prefix-sum to get coverage count at each position
    for i in range(1, N):
        coverage[i] += coverage[i-1]

    # Build result: keep chars not covered by any matched-interval
    out = []
    for i, ch in enumerate(S):
        if coverage[i] == 0:
            out.append(ch)

    sys.stdout.write(''.join(out))

if __name__ == "__main__":
    main()