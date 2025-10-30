def isqrt(n):
    if n == 0:
        return 0
    x = n
    while True:
        y = (x + n // x) // 2
        if y >= x:
            return x
        x = y

N = int(input())
count = 0
power_of_2 = 2
while power_of_2 <= N:
    m_max = isqrt(N // power_of_2)
    count += (m_max + 1) // 2
    power_of_2 *= 2
print(count)