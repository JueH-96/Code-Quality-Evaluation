# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    lcp = 0
    min_len = min(len(S), len(T))
    while lcp < min_len and S[lcp] == T[lcp]:
        lcp +=1

    lcs = 0
    while lcs < min_len - lcp and S[-(lcs+1)] == T[-(lcs+1)]:
        lcs +=1

    S1 = S[lcp:len(S)-lcs if lcs > 0 else len(S)]
    T1 = T[lcp:len(T)-lcs if lcs > 0 else len(T)]

    lenS1 = len(S1)
    lenT1 = len(T1)

    maxD = K
    V = {}
    V[0] = 0
    found = False

    for D in range(0, maxD + 1):
        newV = {}
        for k in range(-D, D+1, 2):
            if k == -D:
                x = V.get(k+1, 0)
            elif k == D:
                x = V.get(k-1, 0) +1
            else:
                if V.get(k-1, -1) +1 < V.get(k+1, -1):
                    x = V.get(k+1, 0)
                else:
                    x = V.get(k-1, 0) +1
            y = x - k

            # Diagonal slide
            while x < lenS1 and y < lenT1 and S1[x] == T1[y]:
                x +=1
                y +=1

            newV[k] = x

            if x >= lenS1 and y >= lenT1:
                found = True
                print("Yes")
                return
        V = newV
    print("No")

threading.Thread(target=main).start()