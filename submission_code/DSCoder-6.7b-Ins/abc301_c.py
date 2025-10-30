import sys

def solve():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    S_count = [0]*26
    T_count = [0]*26
    S_at = T_at = 0

    for s, t in zip(S, T):
        if s == '@':
            S_at += 1
        else:
            S_count[ord(s) - ord('a')] += 1

        if t == '@':
            T_at += 1
        else:
            T_count[ord(t) - ord('a')] += 1

    for i in range(26):
        if S_count[i] > T_count[i] + T_at:
            print("No")
            return

    print("Yes")

solve()