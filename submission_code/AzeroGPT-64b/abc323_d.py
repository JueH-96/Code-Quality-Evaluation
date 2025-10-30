import heapq
import sys
input = sys.stdin.readline

def ilist(): return list(map(int, input().split()))
def mlist(): return map(int, input().split())

N = int(input())
A = [ilist() for _ in range(N)]
A.sort(key=lambda x: x[0])
S, C = zip(*A)
S = list(S)

ans = 0
heapqA = [(c, s) for (s, c) in A]
heapq.heapify(heapqA)

while True:
    if len(heapqA) == 1:
        ans += 1
        break

    c0, s0 = heapq.heappop(heapqA)
    if not c0:
        continue

    c_sum = c0
    while True:
        if len(heapqA) == 0:
            ans += c_sum
            if c_sum % 2 != 0:
                ans += 1
            break

        c1, s1 = heapq.heappop(heapqA)
        if not c1:
            continue
        c_sum += c1
        if s1 != s0 * 2:
            heapq.heappush(heapqA, (c1, s1))
            break

    if c_sum % 2 != 0:
        ans += 1
print(ans)