import sys

def main() -> None:
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    pos = []                # stores 1-indexed positions of correct characters
    i = 0                   # pointer in S

    for j, ch in enumerate(T):   # j is 0-indexed position in T
        if i < len(S) and ch == S[i]:
            pos.append(j + 1)    # convert to 1-indexed
            i += 1
            if i == len(S):      # all characters of S have been found
                break

    print(' '.join(map(str, pos)))

if __name__ == "__main__":
    main()