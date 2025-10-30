import sys

def main() -> None:
    data = list(map(int, sys.stdin.read().split()))
    if not data:                 # no input (should not happen on the judge)
        return

    N, K = data[0], data[1]
    R = data[2:2 + N]

    seq = [0] * N                # current sequence under construction
    out_lines = []               # collect the answer lines here

    # depth-first search that builds the sequence in lexicographical order
    def dfs(pos: int, rem: int) -> None:
        """
        pos : current position (0-based index in the sequence)
        rem : current sum modulo K
        """
        if pos == N:             # sequence completed
            if rem % K == 0:     # sum divisible by K
                out_lines.append(' '.join(map(str, seq)))
            return

        # iterate through possible values for this position in ascending order
        for val in range(1, R[pos] + 1):
            seq[pos] = val
            dfs(pos + 1, (rem + val) % K)

    dfs(0, 0)

    # print all valid sequences (nothing is printed if out_lines is empty)
    sys.stdout.write('
'.join(out_lines))

if __name__ == "__main__":
    main()