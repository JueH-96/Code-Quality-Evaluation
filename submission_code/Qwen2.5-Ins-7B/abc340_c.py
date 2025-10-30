# YOUR CODE HERE
def total_payment(N):
    total = 0
    while N >= 2:
        if N % 2 == 0:
            total += N
            N //= 2
            total += N
            N //= 2
        else:
            total += N
            N = (N // 2) + 1
            total += N
            N = N // 2
    return total

N = int(input())
print(total_payment(N))