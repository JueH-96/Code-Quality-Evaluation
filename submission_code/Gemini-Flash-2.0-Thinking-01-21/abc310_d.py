def solve():
    n, t, m = map(int, input().split())
    incompatible_pairs = []
    for _ in range(m):
        u, v = map(int, input().split())
        incompatible_pairs.append((u, v))
    
    memo = {}
    
    def is_valid_assignment(assignment):
        for pair in incompatible_pairs:
            player1, player2 = pair
            if assignment[player1-1] == assignment[player2-1]:
                return False
        team_counts = [0] * t
        for team_id in assignment:
            team_counts[team_id-1] += 1
        for count in team_counts:
            if count == 0:
                return False
        return True
        
    def count_valid_divisions(player_index, current_assignment):
        if player_index > n:
            if is_valid_assignment(current_assignment):
                return 1
            else:
                return 0
        
        state = (player_index, tuple(current_assignment))
        if state in memo:
            return memo[state]
        
        count = 0
        for team_id in range(1, t + 1):
            next_assignment = current_assignment + [team_id]
            count += count_valid_divisions(player_index + 1, next_assignment)
            
        memo[state] = count
        return count
        
    result = count_valid_divisions(1, [])
    print(result)

if __name__ == '__main__':
    solve()