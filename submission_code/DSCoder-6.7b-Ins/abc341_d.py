def kth_number(N, M, K):
    i = 1
    count = 0
    while True:
        if i % N == 0 or i % M == 0:
            count += 1
        if count == K:
            return i
        i += 1

N, M, K = map(int, input().split())
print(kth_number(N, M, K))