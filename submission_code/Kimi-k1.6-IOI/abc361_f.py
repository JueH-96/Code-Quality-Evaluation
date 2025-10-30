import sys

memo = {}

def is_less_or_equal(mid, b, N):
    res = 1
    for _ in range(b):
        res *= mid
        if res > N:
            return False
    return True

def find_a_max(N, b):
    if b == 0:
        return 1 if N >= 1 else 0
    low = 1
    high = N
    best = 0
    while low <= high:
        mid = (low + high) // 2
        if is_less_or_equal(mid, b, N):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

def count_perfect_powers(M):
    if M < 1:
        return 0
    if M == 1:
        return 1
    if M in memo:
        return memo[M]
    
    low = 1
    high = 60
    max_b = 0
    while low <= high:
        mid = (low + high) // 2
        if (1 << mid) <= M:
            max_b = mid
            low = mid + 1
        else:
            high = mid - 1
    
    total = 1  # x=1
    for b in range(max_b, 1, -1):
        a_max = find_a_max(M, b)
        if a_max < 2:
            continue
        cnt = count_perfect_powers(a_max)
        total += (a_max - cnt)
    
    memo[M] = total
    return total

N = int(sys.stdin.readline())
print(count_perfect_powers(N))