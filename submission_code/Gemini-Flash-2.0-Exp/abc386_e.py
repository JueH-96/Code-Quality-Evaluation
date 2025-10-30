from itertools import combinations

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    max_xor = 0
    for comb in combinations(a, k):
        xor_sum = 0
        for num in comb:
            xor_sum ^= num
        max_xor = max(max_xor, xor_sum)
    
    print(max_xor)

solve()