from itertools import accumulate
from bisect import bisect_right
from operator import itemgetter

def solve(N, X, Y, AB):
    dishes = sorted(enumerate(AB), key=lambda x: x[1][0] + x[1][1])
    sweetness = list(accumulate((a for i, (a, b) in dishes), initial=0))
    saltiness = list(accumulate((b for i, (a, b) in dishes), initial=0))
    ans = 0
    for i, (a, b) in dishes:
        x = bisect_right(sweetness, X - a)
        y = bisect_right(saltiness, Y - b)
        ans = max(ans, min(x, y))
    return ans

N, X, Y = map(int, input().split())
AB = [list(map(int, input().split())) for _ in range(N)]
print(solve(N, X, Y, AB))