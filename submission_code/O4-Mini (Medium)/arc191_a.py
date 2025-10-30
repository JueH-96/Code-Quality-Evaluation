import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    S_str = next(it).strip()
    T_str = next(it).strip()

    # Convert S to list of ints
    S = [ord(c) - 48 for c in S_str]

    # Count digits in T
    cntT = [0] * 10
    for c in T_str:
        cntT[ord(c) - 48] += 1
    # Save initial count of the last T digit
    TL = ord(T_str[-1]) - 48
    init_cnt_TL = cntT[TL]

    # Greedy replacement with all T digits (we will force-use TL later if needed)
    res = [0] * N
    for i in range(N):
        si = S[i]
        # find largest d > si with cntT[d] > 0
        placed = False
        # digits 9 down to si+1
        for d in range(9, si, -1):
            if cntT[d] > 0:
                res[i] = d
                cntT[d] -= 1
                placed = True
                break
        if not placed:
            res[i] = si

    # Check if we used at least one TL in greedy
    used_TL = (init_cnt_TL - cntT[TL]) > 0

    # If TL was never used, we must force one placement of TL
    if not used_TL:
        p = -1
        # 1) beneficial placement: TL > res[i]? pick smallest i
        for i in range(N):
            if TL > res[i]:
                p = i
                break
        if p == -1:
            # 2) neutral placement: find a pos where res[i] == TL, pick rightmost
            for i in range(N-1, -1, -1):
                if res[i] == TL:
                    p = i
                    break
        if p == -1:
            # 3) minimal loss: res[i] > TL; pick minimal (res[i]-TL), tie -> rightmost
            min_loss = 10
            for i in range(N):
                if res[i] > TL:
                    loss = res[i] - TL
                    if loss < min_loss:
                        min_loss = loss
                        p = i
                    elif loss == min_loss:
                        # rightmost of same loss
                        p = i
        # Perform forced replacement
        if p >= 0:
            res[p] = TL
        # else: theoretically impossible (we always can place TL)

    # Output the result
    out = ''.join(chr(d + 48) for d in res)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()