from itertools import combinations

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    max_xor = 0
    for comb in combinations(a, k):
        current_xor = 0
        for x in comb:
            current_xor ^= x
        max_xor = max(max_xor, current_xor)
    
    print(max_xor)

solve()