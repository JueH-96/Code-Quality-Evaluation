def solve():
    n = int(input())
    d = list(map(int, input().split()))
    l1, c1, k1 = map(int, input().split())
    l2, c2, k2 = map(int, input().split())

    total_cost = 0
    for section_len in d:
        min_cost_for_section = float('inf')
        for num_sensor1 in range(k1 + 1):
            for num_sensor2 in range(k2 + 1):
                if num_sensor1 * l1 + num_sensor2 * l2 >= section_len:
                    cost = num_sensor1 * c1 + num_sensor2 * c2
                    min_cost_for_section = min(min_cost_for_section, cost)
        if min_cost_for_section == float('inf'):
            print("-1")
            return
        total_cost += min_cost_for_section
    print(total_cost)

solve()