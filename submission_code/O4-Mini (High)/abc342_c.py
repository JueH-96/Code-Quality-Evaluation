import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    s = next(it)
    q = int(next(it))

    # f[x] will store the current character (0-25) that original char x maps to.
    f = list(range(26))

    # Process each operation by remapping all f[x] == c to d.
    for _ in range(q):
        c = ord(next(it)) - 97
        d = ord(next(it)) - 97
        if c != d:
            for x in range(26):
                if f[x] == c:
                    f[x] = d

    # Build the resulting string.
    out = []
    for ch in s:
        out.append(chr(f[ord(ch) - 97] + 97))

    sys.stdout.write(''.join(out))


if __name__ == "__main__":
    main()