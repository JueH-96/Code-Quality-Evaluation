def solve():
    n = int(input())
    s = input()
    c = list(map(int, input().split()))

    min_cost = float('inf')

    def is_good(t):
        count = 0
        for i in range(n - 1):
            if t[i] == t[i + 1]:
                count += 1
        return count == 1

    def calculate_cost(initial_s, target_s, costs):
        cost = 0
        for i in range(n):
            if initial_s[i] != target_s[i]:
                cost += costs[i]
        return cost

    for i in range(1 << n):
        temp_s_list = list(s)
        current_cost = 0
        for j in range(n):
            if (i >> j) & 1:
                current_cost += c[j]
                temp_s_list[j] = '1' if temp_s_list[j] == '0' else '0'

        transformed_s = "".join(temp_s_list)
        if is_good(transformed_s):
            min_cost = min(min_cost, current_cost)

    print(min_cost)

solve()