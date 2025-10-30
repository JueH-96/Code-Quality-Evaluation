from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        intervals.sort()
        p, q, r, s, a, b, c, d, A, B, C, D = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        p1, q1, r1, s1, a1, b1, c1, d1, A1, B1, C1, D1 = -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        for i, (l, r, w) in enumerate(intervals):
            if l > r:
                continue
            if l > r1:
                s,  s1 = w,  i
                r,  r1 = s,  s1
                q,  q1 = r,  r1
                p,  p1 = q,  q1
            elif l > q1:
                r,  r1 = w,  i
                q,  q1 = r,  r1
                p,  p1 = q,  q1
            elif l > p1:
                q,  q1 = w,  i
                p,  p1 = q,  q1
            if  p + l > c1:
                d,  d1 = p + w, p1 + [i]
                c,  c1 = d,  d1
                b,  b1 = c,  c1
                a,  a1 = b,  b1
            elif p + l > b1:
                c,  c1 = p + w, p1 + [i]
                b,  b1 = c,  c1
                a,  a1 = b,  b1
            elif p + l > a1:
                b,  b1 = p + w, p1 + [i]
                a,  a1 = b,  b1
            if  q + l > C1:
                D,  D1 = q + w, q1 + [i]
                C,  C1 = D,  D1
                B,  B1 = C,  C1
                A,  A1 = B,  B1
            elif q + l > B1:
                C,  C1 = q + w, q1 + [i]
                B,  B1 = C,  C1
                A,  A1 = B,  B1
            elif q + l > A1:
                B,  B1 = q + w, q1 + [i]
                A,  A1 = B,  B1
        ret = A1 if A1[0] > a1[0] else a1
        return ret[:min(len(ret), 4)]