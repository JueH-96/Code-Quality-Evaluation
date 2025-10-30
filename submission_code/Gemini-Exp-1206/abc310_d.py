def solve():
    n, t, m = map(int, input().split())
    incompatible_pairs = []
    for _ in range(m):
        incompatible_pairs.append(tuple(map(int, input().split())))

    count = 0
    
    def is_valid(division):
        for pair in incompatible_pairs:
            team1 = -1
            team2 = -1
            for i in range(t):
                if pair[0] in division[i]:
                    team1 = i
                if pair[1] in division[i]:
                    team2 = i
            if team1 == team2 and team1 != -1:
                return False
        return True

    def backtrack(index, current_division):
        nonlocal count
        if index == n + 1:
            if all(len(team) > 0 for team in current_division) and is_valid(current_division):
                count += 1
            return

        for i in range(t):
            current_division[i].append(index)
            backtrack(index + 1, current_division)
            current_division[i].pop()

    initial_division = [[] for _ in range(t)]
    backtrack(1, initial_division)
    print(count)

solve()