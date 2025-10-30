import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    N = int(data[0])                 # length of the string (not actually needed)
    S = data[1].decode()             # original string
    Q = int(data[2])                 # number of operations

    # mapping[x] is the current letter (0..25) to which the
    # original letter with index x is transformed after the
    # operations processed so far
    mapping = list(range(26))

    pos = 3                          # position in the token list
    for _ in range(Q):
        c = data[pos][0]             # byte value of c_i  (0..255)
        d = data[pos + 1][0]         # byte value of d_i
        pos += 2
        c_idx = c - 97               # 0..25
        d_idx = d - 97

        # update every letter that is currently mapped to c_idx
        for j in range(26):
            if mapping[j] == c_idx:
                mapping[j] = d_idx

    # build the final string
    result = ''.join(chr(mapping[ord(ch) - 97] + 97) for ch in S)
    sys.stdout.write(result)


if __name__ == "__main__":
    main()