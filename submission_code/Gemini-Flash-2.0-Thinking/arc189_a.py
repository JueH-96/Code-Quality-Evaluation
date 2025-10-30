def solve():
    n = int(input())
    target = list(map(int, input().split()))
    initial = [(i % 2) for i in range(1, n + 1)]
    mod = 998244353

    memo = {}

    def find_operations(current_grid):
        current_tuple = tuple(current_grid)
        if current_tuple in memo:
            return memo[current_tuple]

        if current_grid == target:
            return 1

        count = 0
        for l in range(n):
            for r in range(l + 2, n):
                if current_grid[l] == current_grid[r]:
                    can_operate = True
                    for i in range(l + 1, r):
                        if current_grid[i] == current_grid[l]:
                            can_operate = False
                            break
                    if can_operate:
                        next_grid = list(current_grid)
                        for i in range(l + 1, r):
                            next_grid[i] = current_grid[l]
                        count = (count + find_operations(next_grid)) % mod
        memo[current_tuple] = count
        return count

    print(find_operations(initial))

solve()