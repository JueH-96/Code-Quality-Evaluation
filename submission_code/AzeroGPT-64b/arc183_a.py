import math

def get_permutation(n, k, pos):
    fact = [math.factorial(i) for i in range(n + k + 1)]
    perm = []
    index = [0] * (n + 1)
    for i in range(1, n + 1):
        perm.append(i)
        index[i] = len(perm) - 1
    result = []
    
    while len(perm) > 0:
        if k >= len(perm):
            for i in perm:
                result.append(i)
            break
        val = math.ceil(pos / fact[k])
        val -= math.ceil(pos / fact[k+1])
        result.append(perm[val])
        perm.pop(val)
        k -= 1
        n -= 1
        pos -= val * fact[k]
    
    return result

n, k = map(int, input().split())

mid = 1 / 2 * math.factorial(n * k) / math.factorial(k)**n
mid = math.ceil(mid)
result = get_permutation(n, k, mid)
result = (result * k)[:n * k]
print(" ".join(map(str, result)))