def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N, Q = map(int, data[0].split())
    l = 1
    r = 2
    total = 0
    index = 1
    for _ in range(Q):
        parts = data[index].split()
        index += 1
        H = parts[0]
        T = int(parts[1])
        if H == 'L':
            if T == l:
                moves = 0
            else:
                d_clock = (T - l) % N
                if d_clock == 0:
                    d_clock = N
                blocked = False
                for k in range(1, d_clock):
                    node = (l - 1 + k) % N + 1
                    if node == r:
                        blocked = True
                        break
                if not blocked:
                    moves = d_clock
                else:
                    moves = N - d_clock
            total += moves
            l = T
        else:
            if T == r:
                moves = 0
            else:
                d_clock = (T - r) % N
                if d_clock == 0:
                    d_clock = N
                blocked = False
                for k in range(1, d_clock):
                    node = (r - 1 + k) % N + 1
                    if node == l:
                        blocked = True
                        break
                if not blocked:
                    moves = d_clock
                else:
                    moves = N - d_clock
            total += moves
            r = T
    print(total)

if __name__ == "__main__":
    main()