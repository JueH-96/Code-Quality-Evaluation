N, M, P = map(int, input().split())
if M > N:
    print(0)
else:
    count = 1
    current = M
    while current + P <= N:
        current += P
        count += 1
    print(count)