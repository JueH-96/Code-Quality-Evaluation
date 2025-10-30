import sys
import threading

def main():
    import sys

    S = sys.stdin.readline().strip()
    if not S:
        return

    # R is the reverse of S
    R = S[::-1]
    # We want the longest prefix of R that matches a suffix of S.
    # Build KMP table on pattern = R + '#' + S
    sep = '#'
    U = R + sep + S
    n = len(U)
    pi = [0] * n
    # compute prefix function
    for i in range(1, n):
        j = pi[i-1]
        while j > 0 and U[i] != U[j]:
            j = pi[j-1]
        if U[i] == U[j]:
            j += 1
        pi[i] = j

    # pi[-1] is the length L of the longest prefix of R matching a suffix of S
    L = pi[-1]
    # We append R[L:] to S to make the palindrome
    ans = S + R[L:]
    sys.stdout.write(ans)

if __name__ == "__main__":
    main()