import sys

def solve():
    N, X, Y = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Condition 1: Total count of 0s and 1s must match
    if S.count('0') != T.count('0'):
        print("No")
        return

    # If N < X + Y, no operations are possible
    if N < X + Y:
        if S == T:
            print("Yes")
        else:
            print("No")
        return

    # Function to check if S[start:start+length] consists of char * length
    def check_pattern(string, start, char, length):
        if start + length > len(string):
            return False
        for i in range(length):
            if string[start + i] != char:
                return False
        return True

    # Condition 2: If S[0] != T[0], check if S[0] can be flipped
    if S[0] != T[0]:
        # S[0]=0, T[0]=1: needs 0 -> 1 flip. Possible only by Op A at i=0.
        # Op A needs pattern 0^X 1^Y at i=0.
        # This requires S[0...X-1] == 0^X and S[X...X+Y-1] == 1^Y
        if S[0] == '0':
            if not (check_pattern(S, 0, '0', X) and check_pattern(S, X, '1', Y)):
                print("No")
                return
        # S[0]=1, T[0]=0: needs 1 -> 0 flip. Possible only by Op B at i=0.
        # Op B needs pattern 1^Y 0^X at i=0.
        # This requires S[0...Y-1] == 1^Y and S[Y...X+Y-1] == 0^X
        else: # S[0] == '1'
            if not (check_pattern(S, 0, '1', Y) and check_pattern(S, Y, '0', X)):
                print("No")
                return

    # Condition 3: If S[N-1] != T[N-1], check if S[N-1] can be flipped
    # Operation window starts at i = N - (X+Y). Window is [N-X-Y, N-1].
    i = N - (X+Y)
    if S[N-1] != T[N-1]:
        # S[N-1]=0, T[N-1]=1: needs 0 -> 1 flip. Possible only by Op B at i=N-(X+Y).
        # Op B needs pattern 1^Y 0^X at i=N-(X+Y).
        # This requires S[i...i+Y-1] == 1^Y and S[i+Y...i+X+Y-1] == 0^X
        # i+X+Y-1 = N-X-Y+X+Y-1 = N-1
        # i+Y = N-X
        # i+Y-1 = N-X-1
        # So check S[N-X-Y...N-X-1] == 1^Y and S[N-X...N-1] == 0^X
        if S[N-1] == '0':
            if not (check_pattern(S, i, '1', Y) and check_pattern(S, i+Y, '0', X)):
                print("No")
                return
        # S[N-1]=1, T[N-1]=0: needs 1 -> 0 flip. Possible only by Op A at i=N-(X+Y).
        # Op A needs pattern 0^X 1^Y at i=N-(X+Y).
        # This requires S[i...i+X-1] == 0^X and S[i+X...i+X+Y-1] == 1^Y
        # i+X = N-Y
        # i+X-1 = N-Y-1
        # So check S[N-X-Y...N-Y-1] == 0^X and S[N-Y...N-1] == 1^Y
        else: # S[N-1] == '1'
             if not (check_pattern(S, i, '0', X) and check_pattern(S, i+X, '1', Y)):
                print("No")
                return

    # If all checks pass, it is possible
    print("Yes")

solve()