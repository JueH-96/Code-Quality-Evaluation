N = int(input())
A = list(map(int, input().split()))
Q = int(input())
for _ in range(Q):
    l, r = map(int, input().split())
    total_sleep_time = sum(A[i] for i in range(l, r+1) if i % 2 == 0)
    print(total_sleep_time)