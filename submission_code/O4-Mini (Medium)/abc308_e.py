import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    S = data[1+N].strip()

    # Precompute mex for triples (x,y,z) where each in {0,1,2}
    mex3 = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                seen0 = False
                seen1 = False
                seen2 = False
                if x == 0 or y == 0 or z == 0:
                    seen0 = True
                if x == 1 or y == 1 or z == 1:
                    seen1 = True
                if x == 2 or y == 2 or z == 2:
                    seen2 = True
                if not seen0:
                    m = 0
                elif not seen1:
                    m = 1
                elif not seen2:
                    m = 2
                else:
                    m = 3
                mex3[x][y][z] = m

    # Build suffix counts of X's by value
    # suffix0[i] = count of positions k>=i with S[k]=='X' and A[k]==0, etc.
    suffix0 = [0] * (N+1)
    suffix1 = [0] * (N+1)
    suffix2 = [0] * (N+1)
    for i in range(N-1, -1, -1):
        suffix0[i] = suffix0[i+1]
        suffix1[i] = suffix1[i+1]
        suffix2[i] = suffix2[i+1]
        if S[i] == 'X':
            ai = A[i]
            if ai == 0:
                suffix0[i] += 1
            elif ai == 1:
                suffix1[i] += 1
            else:
                suffix2[i] += 1

    # Prefix counts of M's by value
    pref = [0, 0, 0]

    ans = 0
    # Iterate positions j; when S[j]=='E', combine
    for j in range(N):
        if S[j] == 'E':
            y = A[j]
            # for each possible x in M's and z in X's
            c0 = pref[0]
            c1 = pref[1]
            c2 = pref[2]
            # suffix counts at j+1 (positions k>j)
            x0 = suffix0[j+1]
            x1 = suffix1[j+1]
            x2 = suffix2[j+1]
            # sum over 3x3 possibilities
            if c0 and x0:
                ans += c0 * x0 * mex3[0][y][0]
            if c0 and x1:
                ans += c0 * x1 * mex3[0][y][1]
            if c0 and x2:
                ans += c0 * x2 * mex3[0][y][2]
            if c1 and x0:
                ans += c1 * x0 * mex3[1][y][0]
            if c1 and x1:
                ans += c1 * x1 * mex3[1][y][1]
            if c1 and x2:
                ans += c1 * x2 * mex3[1][y][2]
            if c2 and x0:
                ans += c2 * x0 * mex3[2][y][0]
            if c2 and x1:
                ans += c2 * x1 * mex3[2][y][1]
            if c2 and x2:
                ans += c2 * x2 * mex3[2][y][2]
        # After handling E, update prefix if M
        if S[j] == 'M':
            pref[A[j]] += 1

    # Print the result
    print(ans)

if __name__ == "__main__":
    main()