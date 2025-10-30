def solve():
    n, w = map(int, input().split())
    items = []
    for _ in range(n):
        items.append(list(map(int, input().split())))

    dp = {}

    def get_happiness(item_type, count):
        return count * items[item_type][1] - count * count

    def get_dp(capacity, item_type):
        if (capacity, item_type) in dp:
            return dp[(capacity, item_type)]

        if item_type == n:
            return 0

        max_happiness = 0
        for count in range(min(capacity // items[item_type][0] + 1, 3001)):
            current_happiness = get_happiness(item_type, count)
            remaining_capacity = capacity - count * items[item_type][0]
            
            max_happiness = max(max_happiness, current_happiness + get_dp(remaining_capacity, item_type + 1))

        dp[(capacity, item_type)] = max_happiness
        return max_happiness

    print(get_dp(w, 0))

solve()