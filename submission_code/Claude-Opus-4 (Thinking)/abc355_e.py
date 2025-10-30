def trailing_zeros(n):
    if n == 0:
        return 64  # A large number, bigger than any i we'll use
    count = 0
    while n % 2 == 0:
        n //= 2
        count += 1
    return count

def floor_log2(n):
    if n <= 0:
        return -1
    log = 0
    while (1 << (log + 1)) <= n:
        log += 1
    return log

def query(i, j):
    print(f"? {i} {j}", flush=True)
    response = int(input())
    if response == -1:
        exit()
    return response

def solve(N, L, R):
    total = 0
    pos = L
    
    while pos <= R:
        # Find the largest i such that:
        # 1. pos is divisible by 2^i
        # 2. pos + 2^i - 1 <= R
        
        tz = trailing_zeros(pos)
        max_len_log = floor_log2(R - pos + 1)
        i = min(tz, max_len_log)
        
        j = pos // (1 << i)
        result = query(i, j)
        total = (total + result) % 100
        
        pos += (1 << i)
    
    print(f"! {total}", flush=True)

N, L, R = map(int, input().split())
solve(N, L, R)