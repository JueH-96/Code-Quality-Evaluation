def compute_mobius(n):
    if n == 1:
        return 1
    factors = {}
    temp = n
    i = 2
    while i * i <= temp:
        if temp % i == 0:
            count = 0
            while temp % i == 0:
                count += 1
                temp = temp // i
            if count > 1:
                return 0
            factors[i] = 1
        i += 1
    if temp > 1:
        factors[temp] = 1
    return (-1) ** len(factors)

def find_a(N, b):
    low = 1
    high = N
    best = 1
    while low <= high:
        mid = (low + high) // 2
        product = 1
        overflow = False
        for _ in range(b):
            product *= mid
            if product > N:
                overflow = True
                break
        if overflow:
            high = mid - 1
        else:
            best = mid
            low = mid + 1
    return best

N = int(input())
if N < 1:
    print(0)
else:
    answer = 1
    max_b = 1
    while (2 ** (max_b + 1)) <= N:
        max_b += 1
    sum_terms = 0
    for b in range(2, max_b + 1):
        mu = compute_mobius(b)
        if mu == 0:
            continue
        a = find_a(N, b)
        if a < 2:
            continue
        sum_terms += mu * (a - 1)
    count = -sum_terms
    answer += count
    print(answer)