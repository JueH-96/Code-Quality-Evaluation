def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return a * b // gcd(a, b)

def compute_count(x, coins):
    n = len(coins)
    count = 0
    for mask in range(1, 1 << n):
        bits = bin(mask).count('1')
        current_lcm = 1
        for i in range(n):
            if mask & (1 << i):
                current_lcm = lcm(current_lcm, coins[i])
        term = x // current_lcm
        if bits % 2 == 1:
            count += term
        else:
            count -= term
    return count

def findKthSmallest(coins, k):
    coins.sort()
    low = coins[0]
    max_coin = coins[-1]
    high = max_coin * k

    while low < high:
        mid = (low + high) // 2
        cnt = compute_count(mid, coins)
        if cnt >= k:
            high = mid
        else:
            low = mid + 1
    return low