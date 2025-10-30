import sys
import threading

def main():
    import sys
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N, Q = map(int, line)
    S = data.readline().strip()
    # Build prefix array of equal adjacent counts
    # pref[i] = number of positions p < i where S[p] == S[p+1], using 0-based indices
    pref = [0] * N
    for i in range(1, N):
        pref[i] = pref[i-1] + (1 if S[i] == S[i-1] else 0)
    out = []
    for _ in range(Q):
        l_str, r_str = data.readline().split()
        l = int(l_str) - 1
        r = int(r_str) - 1
        # The count of equal adjacent letters in S[l..r] is pref[r] - pref[l]
        out.append(str(pref[r] - pref[l]))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()