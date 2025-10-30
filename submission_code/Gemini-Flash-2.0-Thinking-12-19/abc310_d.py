import collections

def solve():
    n, t, m = map(int, input().split())
    incompatible_pairs = []
    for _ in range(m):
        u, v = map(int, input().split())
        incompatible_pairs.append(tuple(sorted((u, v))))
    
    memo = {}
    
    def count_valid_assignments(player_index, current_team_assignments, num_teams):
        if player_index > n:
            return 1
        state = (player_index,) + tuple(current_team_assignments)
        if state in memo:
            return memo[state]
        
        count = 0
        for team_id in range(1, num_teams + 1):
            is_valid_team = True
            for i in range(1, player_index):
                pair = tuple(sorted((i, player_index)))
                if pair in incompatible_pairs:
                    if current_team_assignments[i-1] == team_id:
                        is_valid_team = False
                        break
            if is_valid_team:
                current_team_assignments[player_index-1] = team_id
                count += count_valid_assignments(player_index + 1, current_team_assignments, num_teams)
                
        memo[state] = count
        return count
        
    def get_n_valid_assignments(num_teams):
        if num_teams <= 0:
            return 0 if n > 0 else 1
        memo.clear()
        return count_valid_assignments(1, [0] * n, num_teams)
        
    def nCr(n_val, r_val):
        if r_val < 0 or r_val > n_val:
            return 0
        if r_val == 0 or r_val == n_val:
            return 1
        if r_val > n_val // 2:
            r_val = n_val - r_val
        result = 1
        for i in range(r_val):
            result = result * (n_val - i) // (i + 1)
        return result
        
    total_ways = 0
    for k in range(t + 1):
        n_assignments = get_n_valid_assignments(t - k)
        combinations = nCr(t, k)
        term = combinations * n_assignments
        if k % 2 == 1:
            total_ways -= term
        else:
            total_ways += term
            
    print(total_ways)

if __name__ == '__main__':
    solve()