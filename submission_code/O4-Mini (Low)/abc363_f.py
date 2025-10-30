import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10000)
    N = int(sys.stdin.readline().strip())
    
    # quick digit checks
    def nod0(s):
        return '0' not in s
    
    def ispal(s):
        return s == s[::-1]
    
    sN = str(N)
    # 1) trivial: decimal palindrome, no zero
    if nod0(sN) and ispal(sN):
        print(sN)
        return
    
    # compute divisors up to sqrt(N)
    import math
    D = []
    lim = int(math.isqrt(N)) + 1
    for i in range(1, lim):
        if N % i == 0:
            D.append(i)
            if i != N//i:
                D.append(N//i)
    # 2) two‐factor palindrome: d * rev(d)
    for d in D:
        sd = str(d)
        if not nod0(sd): continue
        revd = sd[::-1]
        jd = int(revd)
        if d * jd == N and nod0(revd):
            print(sd + "*" + revd)
            return
    
    # 3) three‐factor form: d * M * rev(d), where M palindrome numeric, no zeros
    # i.e. N = d^2 * M
    for d in D:
        sd = str(d)
        if not nod0(sd): continue
        revd = sd[::-1]
        if not nod0(revd): continue
        d2 = d*d
        if d2 > N or N % d2 != 0:
            continue
        M = N // d2
        sM = str(M)
        if nod0(sM) and ispal(sM):
            S = sd + "*" + sM + "*" + revd
            if len(S) <= 1000:
                print(S)
                return
    
    # give up
    print(-1)

if __name__ == "__main__":
    main()