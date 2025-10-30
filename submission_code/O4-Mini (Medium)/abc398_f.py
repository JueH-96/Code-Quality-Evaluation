import sys
import threading

def main():
    import sys
    S = sys.stdin.readline().rstrip()
    if not S:
        return
    A = S
    B = S[::-1]
    # We want the longest l such that A's suffix of length l == B's prefix of length l.
    # Build T = B + '#' + A and compute prefix-function.
    T = B + '#' + A
    n = len(T)
    pi = [0] * n
    # KMP prefix-function
    for i in range(1, n):
        j = pi[i-1]
        while j > 0 and T[i] != T[j]:
            j = pi[j-1]
        if T[i] == T[j]:
            j += 1
        pi[i] = j
    l = pi[-1]
    # append the rest of B after the matched prefix
    to_append = B[l:]
    sys.stdout.write(A + to_append)

if __name__ == "__main__":
    main()