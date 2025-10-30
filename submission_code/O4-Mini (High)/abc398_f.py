import sys

def main():
    S = sys.stdin.readline().rstrip('
')
    R = S[::-1]
    # Build the string R + '#' + S and compute its prefix-function (KMP failure array)
    A = R + '#' + S
    m = len(A)
    pi = [0] * m
    for i in range(1, m):
        j = pi[i - 1]
        # fall back while mismatch
        while j > 0 and A[i] != A[j]:
            j = pi[j - 1]
        # if match, extend
        if A[i] == A[j]:
            j += 1
        pi[i] = j

    # pi[-1] is the length of the longest prefix of R matching a suffix of S
    k = pi[-1]
    # append the unmatched part of R to S
    result = S + R[k:]
    sys.stdout.write(result)

# Call main to execute
main()