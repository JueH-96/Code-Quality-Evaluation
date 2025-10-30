N = int(input())
P = list(map(int, input().split()))
Q = int(input())
queries = [list(map(int, input().split())) for _ in range(Q)]

for query in queries:
    A, B = query
    print(min(P[A-1:B]))