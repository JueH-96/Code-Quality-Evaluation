import sys
from collections import Counter
input = sys.stdin.readline
def convexCombinationCounter(a):
    freqs = Counter(a)
    result = 0

    for k, v in freqs.items():
        if v > 1:
            result += v * (v - 1) // 2

    primary = Counter()
    for x in freqs:
        double_log = 2 * x
        ratio = None

        for y in freqs:
            if x == y:
                continue

            if y % double_log == 0:
                ex = y // double_log
                if ex % x == 0:
                    ratio = ex // x
                else:
                    ratio = x / ex

                if ratio is not None:
                    if ratio > 1 and ratio in primary and x in primary[ratio]:
                        result += primary[ratio][x] * freqs[x] * freqs[y]
                    elif ratio < 1 and x in primary and 1 / ratio in primary[x]:
                        result += primary[x][1/ratio] * freqs[x] * freqs[y]

        if 2 * x in freqs:
            if x == 2:
                duals = freqs[4]
                if duals > 0:
                    result += duals * freqs[2] * (freqs[2] - 1) // 2

                if duals > 1:
                    result += duals * (duals - 1) // 2
            else:
                result += freqs[2*x] * (freqs[x] * (freqs[x] - 1) // 2)

        for y in freqs:
            if x == y:
                continue

            ex = x / 2
            if y % ex == 0:
                ratio = y // ex

                if ratio > 1:
                    if ratio in primary:
                        if x in primary[ratio]:
                            primary[ratio][x] += freqs[y]
                        else:
                            primary[ratio][x] = freqs[y]
                    else:
                        primary[ratio] = Counter({x: freqs[y]})
                elif ratio < 1:
                    if x in primary:
                        if 1 / ratio in primary[x]:
                            primary[x][1/ratio] += freqs[y]
                        else:
                            primary[x][1/ratio] = freqs[y]
                    else:
                        primary[x] = Counter({1/ratio: freqs[y]})

    return result

tests = int(input().strip())
for _ in range(tests):
    _ = input().strip()
    print(convexCombinationCounter(list(map(int, input().strip().split()))))