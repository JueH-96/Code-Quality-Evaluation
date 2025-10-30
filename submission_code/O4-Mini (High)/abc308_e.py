import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    S = input().strip()

    # Precompute mex for all triples (a,b,c) in {0,1,2}^3
    f = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for a in range(3):
        for b in range(3):
            for c in range(3):
                present = [False]*4
                present[a] = True
                present[b] = True
                present[c] = True
                m = 0
                while present[m]:
                    m += 1
                f[a][b][c] = m

    # cnt_M[x] = how many 'M' we've seen with value x
    cnt_M = [0]*3
    # cnt_ME[a][b] = how many pairs (i<j) with S_i='M', A_i=a and S_j='E', A_j=b
    cnt_ME = [[0]*3 for _ in range(3)]

    ans = 0
    for i, ch in enumerate(S):
        x = A[i]
        if ch == 'M':
            cnt_M[x] += 1
        elif ch == 'E':
            # extend all existing M->E pairs
            cnt_ME[0][x] += cnt_M[0]
            cnt_ME[1][x] += cnt_M[1]
            cnt_ME[2][x] += cnt_M[2]
        else:  # ch == 'X'
            # close all M->E->X triplets
            c = x
            # unroll a=0,1,2 loops for speed
            row0, row1, row2 = cnt_ME[0], cnt_ME[1], cnt_ME[2]
            f0, f1, f2 = f[0], f[1], f[2]
            ans += row0[0]*f0[0][c] + row0[1]*f0[1][c] + row0[2]*f0[2][c]
            ans += row1[0]*f1[0][c] + row1[1]*f1[1][c] + row1[2]*f1[2][c]
            ans += row2[0]*f2[0][c] + row2[1]*f2[1][c] + row2[2]*f2[2][c]

    print(ans)

if __name__ == '__main__':
    main()