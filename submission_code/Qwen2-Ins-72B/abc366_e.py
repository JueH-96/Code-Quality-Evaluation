import sys
from heapq import heappush, heappop
from collections import defaultdict

def read_ints():
    return list(map(int, input().split()))

def read_int():
    return int(input())

def read_str():
    return input().strip()

def solve():
    N, D = read_ints()
    X = []
    Y = []
    for _ in range(N):
        x, y = read_ints()
        X.append(x)
        Y.append(y)
    
    def count_points(axis, other_axis):
        axis.sort()
        other_axis.sort()
        axis = [0] + axis + [axis[-1]]
        other_axis = [0] + other_axis + [other_axis[-1]]
        n = len(axis)
        m = len(other_axis)
        ans = 0
        h = []
        for i in range(n):
            while h and h[0] < axis[i]:
                j = heappop(h)
                ans += (n - i - 1) * (j + 1)
            while j < m and other_axis[j] - axis[i] <= D:
                heappush(h, other_axis[j])
                j += 1
            ans += (n - i - 1) * (j - len(h))
        return ans
    
    ans = count_points(X, Y) + count_points(Y, X)
    print(ans)

solve()