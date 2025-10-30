import sys
import threading

def main():
    import sys
    data = sys.stdin
    readline = data.readline

    N = int(readline())
    S = list(readline().rstrip('
'))
    Q = int(readline())

    # ltime[i] = the operation‐index of the last type‐1 update on position i (0 if none)
    ltime = [0] * N
    # lastChar[i] = the character assigned at that last type‐1 update (or initial S[i] if none)
    lastChar = S

    # Track only the last global operation (type 2 or 3)
    last_glob_idx = 0
    last_glob_type = 0  # 0 = none, 2 = to lowercase, 3 = to uppercase

    for i in range(1, Q+1):
        line = readline().split()
        t = int(line[0])
        if t == 1:
            x = int(line[1]) - 1
            c = line[2]
            ltime[x] = i
            lastChar[x] = c
        else:
            # type 2 or 3: record as the latest global op
            last_glob_idx = i
            last_glob_type = t

    # Build the final string
    out = []
    lgp = last_glob_idx
    lgt = last_glob_type
    if lgt == 0:
        # No global convert at all
        sys.stdout.write(''.join(lastChar))
        return

    # If there was a global convert, characters whose last type‐1 update
    # happened before that global convert (ltime < lgp) are transformed,
    # others (ltime > lgp) stay as they are.
    if lgt == 2:
        # convert uppercase -> lowercase
        for i in range(N):
            c = lastChar[i]
            if ltime[i] < lgp:
                out.append(c.lower())
            else:
                out.append(c)
    else:
        # lgt == 3, convert lowercase -> uppercase
        for i in range(N):
            c = lastChar[i]
            if ltime[i] < lgp:
                out.append(c.upper())
            else:
                out.append(c)

    sys.stdout.write(''.join(out))


if __name__ == "__main__":
    main()