import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    S = data[1]
    # Stack for matching parentheses
    stack = []
    match = [-1] * N
    for i, ch in enumerate(S):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                l = stack.pop()
                match[l] = i
                match[i] = l

    # Use a difference array to mark intervals to delete
    diff = [0] * (N + 1)
    for i in range(N):
        if S[i] == '(' and match[i] != -1:
            l = i
            r = match[i]
            diff[l] += 1
            diff[r + 1] -= 1

    # Build the result by skipping marked positions
    res = []
    cur = 0
    for i in range(N):
        cur += diff[i]
        if cur == 0:
            res.append(S[i])

    sys.stdout.write("".join(res))

if __name__ == "__main__":
    main()