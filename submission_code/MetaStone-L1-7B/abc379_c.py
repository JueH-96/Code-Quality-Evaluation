def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    X = list(map(int, input[idx:idx+M]))
    idx += M

    X.sort()

    sum_A = sum(X)
    if sum_A != N:
        print(-1)
        return

    prefix = []
    current = 0
    for x in X:
        current += x
        prefix.append(current)
        if current < 1:
            print(-1)
            return

    stones_moved_prev = 0
    total = 0
    for x in X:
        stones_in = x + stones_moved_prev
        moved = max(stones_in - 1, 0)
        if stones_moved_prev > 0 and x == 0:
            print(-1)
            return
        total += moved
        stones_moved_prev = moved

    print(total)

if __name__ == '__main__':
    main()