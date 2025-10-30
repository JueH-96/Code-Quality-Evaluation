class Solution:
    def maxScore(self, a: List[int], b: List[int]) -> int:
        sign = tuple(c > 0 for c in a)
        sorted_b = sorted((x, i) for i, x in enumerate(b))
        by_sign = []
        for s in range(16):
            min_val = [sys.maxsize] * 4
            for x, i in sorted_b:
                bit = (s >> i & 1) * 2 - 1
                for k in range(4):
                    if k == 0 or bit * min_val[k - 1][1] > bit * x:
                        min_val[k] = (x, i)
                    if bit * min_val[k][1] > bit * min_val[k][0]:
                        min_val[k] = (min_val[k][1], i)
            by_sign.append(min_val)
        if sign[1]:
            by_sign.append([min(by_sign[s][i], key=lambda t: t[0] * a[i]) for i in range(4)])
        else:
            by_sign.append([max(by_sign[s][i], key=lambda t: t[0] * a[i]) for i in range(4)])

        # ensure i0 < i1 < i2 < i3
        max_val = []
        for s0, s1 in ((0b0001, 0b1000), (0b0011, 0b1100), (0b0111, 0b1110), (0b1111, 0b1111)):
            max_val.append(
                max(by_sign[s][0][1] < by_sign[(s + s0) & s1][1][1],
                    by_sign[(s + s0) & s1][1][1] < by_sign[(s + s0 + s1) & s1][2][1],
                    by_sign[(s + s0 + s1) & s1][2][1] < by_sign[(s + s0 + s1) ^ s1][3][1])
                for s in range(16))
            by_sign[-1] = [t for t, _ in zip(by_sign[-1], max_val) if _]

        return max(sum(a[i] * x for i, x in enumerate(t))
                   for t in by_sign[-1])