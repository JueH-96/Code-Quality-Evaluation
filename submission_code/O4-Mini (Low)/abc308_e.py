import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    S = data[1+N]

    # cntM[v]: number of 'M' positions seen so far with A == v
    cntM = [0,0,0]
    # countPairs[x][y]: number of pairs (i,j) with i<'E' j processed so far,
    # S[i]=='M' with A_i==x, S[j]=='E' with A_j==y
    countPairs = [[0]*3 for _ in range(3)]

    # Precompute mex for all triples of 0<=x,y,z<=2
    mex3 = [[[0]*3 for _ in range(3)] for __ in range(3)]
    for x in range(3):
        for y in range(3):
            for z in range(3):
                s = {x,y,z}
                m = 0
                while m in s:
                    m += 1
                mex3[x][y][z] = m

    ans = 0

    for i in range(N):
        c = S[i]
        v = A[i]
        if c == 'M':
            # register a new M
            cntM[v] += 1
        elif c == 'E':
            # for each prior M with value x, make pairs (x, v)
            # add cntM[x] to countPairs[x][v]
            for x in range(3):
                cx = cntM[x]
                if cx:
                    countPairs[x][v] += cx
        else:  # c == 'X'
            # for each (x,y) pair, with countPairs[x][y] occurrences,
            # add count * mex(x,y,v)
            for x in range(3):
                row = countPairs[x]
                for y in range(3):
                    cnt = row[y]
                    if cnt:
                        ans += cnt * mex3[x][y][v]

    print(ans)

if __name__ == "__main__":
    main()