import sys
import threading

def main():
    import math
    data = sys.stdin.read().strip()
    if not data:
        print(-1)
        return
    try:
        N = int(data.strip())
    except:
        print(-1)
        return

    # trivial: N itself if palindrome and no zero
    sN = str(N)
    if '0' not in sN and sN == sN[::-1]:
        print(sN)
        return

    # 1-layer: S = a * M * rev(a)
    # try a up to AL1 or sqrt(N)
    sqrtN = int(math.isqrt(N))
    AL1 = min(sqrtN, 20000)
    for a in range(1, AL1+1):
        a_str = str(a)
        if '0' in a_str: 
            continue
        rev_a_str = a_str[::-1]
        rev_a = int(rev_a_str)
        denom = a * rev_a
        if denom == 0 or denom > N:
            continue
        if N % denom != 0:
            continue
        M = N // denom
        M_str = str(M)
        if '0' in M_str:
            continue
        if M_str != M_str[::-1]:
            continue
        # build palindrome expression
        S = a_str + '*' + M_str + '*' + rev_a_str
        print(S)
        return

    # 2-layer: a1 * a2 * M * rev(a2) * rev(a1)
    # try small a1, a2
    BL1 = 100
    BL2 = 10000
    for a1 in range(1, BL1+1):
        a1_str = str(a1)
        if '0' in a1_str:
            continue
        rev1_str = a1_str[::-1]
        rev1 = int(rev1_str)
        d1 = a1 * rev1
        if d1 == 0 or d1 > N:
            continue
        # inner loop
        # we can break early if d1 * (1*1) > N but 1*1 minimal
        for a2 in range(1, BL2+1):
            a2_str = str(a2)
            if '0' in a2_str:
                continue
            rev2_str = a2_str[::-1]
            rev2 = int(rev2_str)
            d2 = a2 * rev2
            denom = d1 * d2
            if denom == 0 or denom > N:
                continue
            if N % denom != 0:
                continue
            M = N // denom
            M_str = str(M)
            if '0' in M_str:
                continue
            if M_str != M_str[::-1]:
                continue
            # build palindrome expression
            # tokens: a1, a2, M, rev(a2), rev(a1)
            S = a1_str + '*' + a2_str + '*' + M_str + '*' + rev2_str + '*' + rev1_str
            print(S)
            return

    # no solution found
    print(-1)

if __name__ == "__main__":
    main()