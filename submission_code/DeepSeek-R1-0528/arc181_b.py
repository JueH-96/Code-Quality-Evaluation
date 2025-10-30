import math

t = int(input().strip())
for _ in range(t):
    S = input().strip()
    X = input().strip()
    Y = input().strip()
    
    n0 = X.count('0')
    n1 = len(X) - n0
    m0 = Y.count('0')
    m1 = len(Y) - m0

    if n0 == m0 and n1 == m1:
        print("Yes")
    elif n1 == m1:
        print("No")
    elif n0 == m0:
        print("Yes")
    else:
        d0 = abs(n0 - m0)
        d1 = abs(n1 - m1)
        L = len(S)
        total_len = d0 * L
        if total_len % d1 != 0:
            print("No")
        else:
            k = total_len // d1
            p = k % L
            if p == 0:
                print("Yes")
            else:
                g = math.gcd(L, p)
                found = True
                for r in range(g):
                    base_char = S[r]
                    j = r
                    while j < L:
                        if S[j] != base_char:
                            found = False
                            break
                        j += g
                    if not found:
                        break
                print("Yes" if found else "No")