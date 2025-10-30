from heapq import heapify, heappush, heappop
from collections import defaultdict
from math import inf
def main():
    N, M = map(int, input().split())
    Roulettes = [list(map(int, input().split())) for _ in range(N)]
    Eps, Ixs = [], []
    for i in range(N):
        C, P, *Ss = Roulettes[i]
        M0 = 0
        CEP = C * P
        for S in Ss:
            M0 += S
            CEP -= S
        # Eps might contain duplicate elements, which can cause issues in 'index' later, so we use a unique identifier.
        # We append (expected value, roulette index) to Eps, and store only roulette indices in Ixs.
        Eps.append((CEP / M0, i))
        Ixs.append(i)

    roulette = defaultdict(int)
    Ev = [inf] * (M + 1)
    Ev[0] = 0
    heapify(Eps)
    for Mth in range(M):
        if Ev[Mth] == inf:
            break
        while True:
            minExpect, minIx = Eps[0]
            if minIx != Ixs[Mth]:
                heappop(Eps)
                continue
            break

        roulette[minIx] += 1
        rouletteE, rouletteCnts, rouletteCnt = [], [], 0
        for _ in range(roulette[minIx]):
            rouletteCnt += 1
            C, P, *Ss = Roulettes[minIx]
            rouletteCnts.append(P)
            rouletteE.append(C)
            for S in Ss:
                rouletteE[-1] -= S
                if rouletteCnts[-1] == 1:
                    Ev[Mth + S] = min(Ev[Mth + S], Ev[Mth] + rouletteE[-1] / sum(Ss))
                elif rouletteCnts[-1] + 1 == rouletteCnt:
                    Ev[Mth] += rouletteE[-1] / sum(Ss)
        while rouletteE:
            rouletteCnt -= rouletteCnts.pop()
            rouletteE.pop()

    sEvM = Ev[M] + rouletteE[-1] / sum(Ss)
    print(sEvM)
if __name__ == "__main__":
    main()