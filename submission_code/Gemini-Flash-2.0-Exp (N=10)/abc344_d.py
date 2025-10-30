def solve():
    T = input()
    N = int(input())
    bags = []
    for _ in range(N):
        line = input().split()
        A = int(line[0])
        bags.append(line[1:])

    dp = {}

    def find_min_cost(current_string, bag_index):
        if current_string == T:
            return 0
        if bag_index == N:
            return float('inf')
        
        if (current_string, bag_index) in dp:
            return dp[(current_string, bag_index)]

        min_cost = find_min_cost(current_string, bag_index + 1)

        for s in bags[bag_index]:
            if T.startswith(current_string + s):
                cost = 1 + find_min_cost(current_string + s, bag_index + 1)
                min_cost = min(min_cost, cost)
        
        dp[(current_string, bag_index)] = min_cost
        return min_cost

    result = find_min_cost("", 0)
    if result == float('inf'):
        print("-1")
    else:
        print(result)

solve()