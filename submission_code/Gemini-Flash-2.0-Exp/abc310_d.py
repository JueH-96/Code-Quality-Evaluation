def solve():
    n, t, m = map(int, input().split())
    incompatible = []
    for _ in range(m):
        a, b = map(int, input().split())
        incompatible.append((a - 1, b - 1))

    count = 0
    
    def check_compatibility(teams):
        for a, b in incompatible:
            team_a = -1
            team_b = -1
            for i in range(len(teams)):
                if a in teams[i]:
                    team_a = i
                if b in teams[i]:
                    team_b = i
            if team_a == team_b:
                return False
        return True

    def generate_teams(index, current_teams):
        nonlocal count
        if index == n:
            if len(current_teams) > t or len(current_teams) == 0:
                return
            
            num_teams = len(current_teams)
            empty_teams = 0
            for team in current_teams:
                if len(team) == 0:
                    empty_teams += 1
            
            if num_teams - empty_teams > t or num_teams - empty_teams < 1:
                return
            
            
            valid_teams = []
            for team in current_teams:
                if len(team) > 0:
                    valid_teams.append(team)
            
            if len(valid_teams) != t:
                return
            
            if check_compatibility(valid_teams):
                count += 1
            return

        # Try adding the current player to an existing team
        for i in range(len(current_teams)):
            new_teams = [team[:] for team in current_teams]
            new_teams[i].append(index)
            generate_teams(index + 1, new_teams)

        # Try creating a new team with the current player
        new_teams = [team[:] for team in current_teams]
        new_teams.append([index])
        generate_teams(index + 1, new_teams)

    generate_teams(0, [[] for _ in range(0)])
    print(count)

solve()