def total_payment(N):
    total = 0
    while N >= 2:
        total += N
        N //= 2
    return total

N = int(input().strip())
print(total_payment(N))