import sys

def prefix_function(s: str) -> list:
    """
    Standard KMP prefix-function.
    pi[i] = length of the longest proper prefix of s[0..i]
            which is also a suffix of s[0..i]
    """
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi


def main() -> None:
    S = sys.stdin.readline().strip()
    if not S:               # in case of an empty line (shouldn’t happen by constraints)
        return

    R = S[::-1]             # reversed string
    T = R + '#' + S         # '#' is a separator not present in S

    pi = prefix_function(T)
    L = pi[-1]              # longest palindromic suffix length

    # Characters to append are the reverse of the prefix that precedes this suffix
    add = S[:len(S) - L][::-1]
    print(S + add)


if __name__ == '__main__':
    main()