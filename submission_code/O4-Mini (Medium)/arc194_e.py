import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N, X, Y = map(int, input().split())
    S = input().strip()
    T = input().strip()

    # Quick total-check: total number of '1's must be equal
    if S.count('1') != T.count('1'):
        print("No")
        return

    # Build runs with token/barrier tags
    # Token A: run of '0' length == X
    # Token B: run of '1' length == Y
    # Barrier: everything else
    def build_runs(st):
        runs = []
        i = 0
        while i < len(st):
            j = i+1
            while j < len(st) and st[j] == st[i]:
                j += 1
            ch = st[i]
            ln = j - i
            if ch == '0' and ln == X:
                runs.append(('A',))  # token A
            elif ch == '1' and ln == Y:
                runs.append(('B',))  # token B
            else:
                runs.append((ch, ln))  # barrier
            i = j
        return runs

    runsS = build_runs(S)
    runsT = build_runs(T)

    # Barriers sequence (char,length) must match in order
    barsS = [(c, l) for (c, *rest) in runsS for l in rest]  # but tokens have rest empty

    # Actually extract barriers list with positions
    def extract_barriers(runs):
        bars = []
        for r in runs:
            if r[0] == 'A' or r[0] == 'B':
                bars.append(None)
            else:
                bars.append(r)
        return bars

    barsS = extract_barriers(runsS)
    barsT = extract_barriers(runsT)
    if len(barsS) != len(barsT):
        print("No")
        return
    for b1, b2 in zip(barsS, barsT):
        if (b1 is None) != (b2 is None):
            print("No")
            return
        if b1 is not None and b2 is not None:
            # barrier must match exactly
            if b1 != b2:
                print("No")
                return

    # Now for each segment between barriers, count tokens
    i = 0
    j = 0
    n = len(runsS)
    m = len(runsT)
    idxS = idxT = 0

    while idxS < n and idxT < m:
        # Skip barrier
        if runsS[idxS][0] != 'A' and runsS[idxS][0] != 'B':
            idxS += 1
            continue
        if runsT[idxT][0] != 'A' and runsT[idxT][0] != 'B':
            idxT += 1
            continue
        # Now both at token segments; count consecutive tokens up to next barrier
        cntA_S = cntB_S = 0
        while idxS < n and runsS[idxS][0] in ('A','B'):
            if runsS[idxS][0] == 'A': cntA_S += 1
            else: cntB_S += 1
            idxS += 1
        cntA_T = cntB_T = 0
        while idxT < m and runsT[idxT][0] in ('A','B'):
            if runsT[idxT][0] == 'A': cntA_T += 1
            else: cntB_T += 1
            idxT += 1
        if cntA_S != cntA_T or cntB_S != cntB_T:
            print("No")
            return

    # If we reach here, all checks passed
    print("Yes")

if __name__ == "__main__":
    main()