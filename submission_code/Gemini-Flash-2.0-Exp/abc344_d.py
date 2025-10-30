def solve():
    T = input()
    N = int(input())
    bags = []
    for _ in range(N):
        line = input().split()
        A = int(line[0])
        strings = line[1:]
        bags.append(strings)

    dp = {}

    def min_cost(index, current_string):
        if current_string == T:
            return 0
        if index == N:
            if current_string == T:
                return 0
            else:
                return float('inf')
        
        if (index, current_string) in dp:
            return dp[(index, current_string)]

        cost1 = min_cost(index + 1, current_string)
        min_c = float('inf')
        for s in bags[index]:
            new_string = current_string + s
            if T.startswith(new_string):
                cost2 = 1 + min_cost(index + 1, new_string)
                min_c = min(min_c, cost2)
        
        dp[(index, current_string)] = min(cost1, min_c)
        return dp[(index, current_string)]

    result = min_cost(0, "")
    if result == float('inf'):
        print("-1")
    else:
        print(result)

solve()