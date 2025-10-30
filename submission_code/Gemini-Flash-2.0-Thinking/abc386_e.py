from itertools import combinations

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    max_xor_sum = 0

    for combo in combinations(a, k):
        current_xor_sum = 0
        for x in combo:
            current_xor_sum ^= x
        max_xor_sum = max(max_xor_sum, current_xor_sum)

    print(max_xor_sum)

solve()