import sys
def main():
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    S = list(input().rstrip())
    # count initial "ABC"
    cnt = 0
    for i in range(N-2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            cnt += 1

    out = []
    for _ in range(Q):
        X, C = input().split()
        idx = int(X) - 1
        # only update if the character actually changes
        if S[idx] != C:
            # remove any "ABC" substrings that included the old character
            for i in range(idx-2, idx+1):
                if 0 <= i <= N-3:
                    if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                        cnt -= 1
            # apply the change
            S[idx] = C
            # add any new "ABC" substrings that include the new character
            for i in range(idx-2, idx+1):
                if 0 <= i <= N-3:
                    if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
                        cnt += 1
        out.append(str(cnt))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()