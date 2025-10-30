import heapq
import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    r.sort()
    queries = [int(sys.stdin.readline()) for _ in range(q)]
    answers = []
    for x in queries:
        sleighs = []
        for i in range(n):
            if r[i] <= x:
                x -= r[i]
                heapq.heappush(sleighs, -r[i])
            else:
                while sleighs and -sleighs[0] > x:
                    x += heapq.heappop(sleighs)
        answers.append(len(sleighs))
    print('
'.join(map(str, answers)))

solve()