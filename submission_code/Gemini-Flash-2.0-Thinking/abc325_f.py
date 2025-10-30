import math

def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())

    dp = {}  # (section_index, num_type1, num_type2) -> min_cost

    def get_dp(section_index, num_type1, num_type2):
        return dp.get((section_index, num_type1, num_type2), float('inf'))

    dp[(0, 0, 0)] = 0

    for i in range(1, n + 1):
        current_d = d[i - 1]
        new_dp = {}
        for k1_used in range(k1 + 1):
            for k2_used in range(k2 + 1):
                min_cost = float('inf')
                for n1 in range(k1_used + 1):
                    for n2 in range(k2_used + 1):
                        if n1 * l1 + n2 * l2 >= current_d:
                            prev_cost = get_dp(i - 1, k1_used - n1, k2_used - n2)
                            if prev_cost != float('inf'):
                                min_cost = min(min_cost, prev_cost + n1 * c1 + n2 * c2)
                if min_cost != float('inf'):
                    new_dp[(i, k1_used, k2_used)] = min_cost
        dp.update(new_dp)

    result = get_dp(n, k1, k2)
    if result == float('inf'):
        print(-1)
    else:
        print(result)

solve()