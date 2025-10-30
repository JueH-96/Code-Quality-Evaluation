N, M = map(int, input().split())
A = list(map(int, input().split()))

j = 0  # pointer to the current firework day
for i in range(1, N+1):
    # Move the pointer until we find a firework day >= i
    while A[j] < i:
        j += 1
    print(A[j] - i)