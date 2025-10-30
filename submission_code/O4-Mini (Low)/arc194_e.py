import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N, X, Y = map(int, data[:3])
    S = data[3].strip()
    T = data[4].strip()
    # 1) counts of '1's must match
    cntS = S.count('1')
    cntT = T.count('1')
    if cntS != cntT:
        print("No")
        return
    # 2) gcd
    from math import gcd
    g = gcd(X, Y)
    # if g==1 then any rearrangement with same counts is reachable
    if g == 1:
        print("Yes")
        return
    # else collect the positions of 1's in S and T
    posS = []
    posT = []
    for i, ch in enumerate(S):
        if ch == '1':
            posS.append(i % g)
    for i, ch in enumerate(T):
        if ch == '1':
            posT.append(i % g)
    # lengths match, now residues must match
    if posS == posT:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()