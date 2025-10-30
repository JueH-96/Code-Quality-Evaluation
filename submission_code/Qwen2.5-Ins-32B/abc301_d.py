import sys

def solve(S, N):
    max_val = int(S.replace('?', '1'), 2)
    if max_val <= N:
        return max_val
    
    min_val = int(S.replace('?', '0'), 2)
    if min_val > N:
        return -1
    
    # Binary search for the maximum value less than or equal to N
    low, high = min_val, max_val
    while low < high:
        mid = (low + high + 1) // 2
        binary_mid = bin(mid)[2:].zfill(len(S))
        if all(a == '?' or a == b for a, b in zip(S, binary_mid)):
            low = mid
        else:
            high = mid - 1
    return low

S = input().strip()
N = int(input().strip())
print(solve(S, N))