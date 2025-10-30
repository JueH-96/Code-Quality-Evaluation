import sys

def minimal_steps(a, b, m):
    if a == b:
        return 0
    # Compute steps for both directions
    diff1 = (b - a) % m
    diff2 = (a - b) % m
    # Check if any direction is possible
    possible = []
    if diff1 != 0:
        possible.append(diff1)
    if diff2 != 0:
        possible.append(diff2)
    if not possible:
        return -1
    return min(possible)

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    A = list(map(int, input[idx:idx+N]))
    idx += N
    B = list(map(int, input[idx:idx+N]))
    idx += N
    total = 0
    for i in range(N):
        a = A[i]
        b = B[i]
        m = M
        if a == b:
            continue
        # Compute steps for both directions
        diff1 = (b - a) % m
        diff2 = (a - b) % m
        # Check if either direction is possible
        possible = []
        if diff1 != 0:
            possible.append(diff1)
        if diff2 != 0:
            possible.append(diff2)
        if not possible:
            print(-1)
            return
        total += min(possible)
    print(total)

if __name__ == '__main__':
    main()