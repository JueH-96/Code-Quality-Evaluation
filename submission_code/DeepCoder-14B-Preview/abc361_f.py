def mobius(n):
    if n == 1:
        return 1
    factors = {}
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            if i in factors:
                return 0
            factors[i] = 1
            while temp % i == 0:
                temp //= i
        i += 1
    if temp > 1:
        if temp in factors:
            return 0
        factors[temp] = 1
    return (-1) ** len(factors)

def max_a(n, k):
    if k == 0:
        return 0
    if n == 0:
        return 0
    low = 2
    high = n
    best = 1
    while low <= high:
        mid = (low + high) // 2
        try:
            power = mid ** k
        except:
            power = float('inf')
        if power == n:
            best = mid
            break
        elif power < n:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

n = int(input())
if n < 1:
    print(0)
    exit()

max_b = 1
while (2 ** (max_b + 1)) <= n:
    max_b += 1

sum_total = 0

for k in range(2, max_b + 1):
    mu = mobius(k)
    a_max = max_a(n, k)
    count = a_max - 1
    if count < 0:
        count = 0
    sum_total += count * (-mu)

total = sum_total + 1
print(total)