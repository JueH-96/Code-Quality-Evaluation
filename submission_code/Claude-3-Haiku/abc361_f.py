import math

def count_powers(N):
    count = 0
    for a in range(2, int(math.sqrt(N)) + 1):
        b = 2
        x = a ** b
        while x <= N:
            count += 1
            b += 1
            x = a ** b
    return count

N = int(input())
print(count_powers(N))