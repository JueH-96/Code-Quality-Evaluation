N = int(input())

def check(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    return n == 1

print('Yes' if check(N) else 'No')