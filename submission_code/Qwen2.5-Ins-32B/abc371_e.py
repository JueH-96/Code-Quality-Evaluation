import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    last_seen = {}
    total = 0
    for i in range(N):
        new_total = 0
        distinct_count = 0
        last_distinct = {}
        for j in range(i, N):
            if A[j] not in last_distinct:
                distinct_count += 1
                last_distinct[A[j]] = True
            new_total += distinct_count
            last_seen[A[j]] = j
        total += new_total
    print(total)

solve()