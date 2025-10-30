import sys
import threading
def main():
    input = sys.stdin.readline
    MOD = 998244353
    T = int(input())
    out = []
    for _ in range(T):
        h, w = map(int, input().split())
        S = [input().strip() for _ in range(h)]
        # Build T[i][j]={1 if A,0 if B}
        # Check T[i][j] ^ T[i][0] ^ T[0][j] ^ T[0][0] == 0 for all i,j
        # If ever nonzero => inconsistent => 0
        ok = True
        # Pre‐compute row0 and col0 first
        # let t00 = T[0][0]
        t00 = 1 if S[0][0]=='A' else 0
        # T[i][0]
        col0 = [1 if S[i][0]=='A' else 0 for i in range(h)]
        # T[0][j]
        row0 = [1 if S[0][j]=='A' else 0 for j in range(w)]
        for i in range(h):
            ti0 = col0[i]
            for j in range(w):
                tij = 1 if S[i][j]=='A' else 0
                # check tij ⊕ ti0 ⊕ row0[j] ⊕ t00 == 0
                if (tij ^ ti0 ^ row0[j] ^ t00) != 0:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            out.append("0")
        else:
            # exactly 2 valid tilings
            out.append("2")
    sys.stdout.write("
".join(out))

if __name__=="__main__":
    main()