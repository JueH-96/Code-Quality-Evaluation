def min_switches_to_equal_strength(N, teams_strength):
    # Calculate the total strength and the strength of each team
    total_strength = sum(strength for _, strength in teams_strength)
    
    # If total strength is not divisible by 3, it's impossible to balance
    if total_strength % 3 != 0:
        return -1
    
    target_strength = total_strength // 3
    
    # Calculate current strengths of each team
    team_strengths = [0, 0, 0]
    for team, strength in teams_strength:
        team_strengths[team - 1] += strength
    
    # Calculate the excess or deficit for each team
    excess = [team_strengths[i] - target_strength for i in range(3)]
    
    # If all teams are already balanced
    if all(e == 0 for e in excess):
        return 0
    
    # Count how many switches are needed
    switches = 0
    
    # We can only switch people from teams with excess to teams with deficit
    while True:
        # Find teams with excess and deficit
        excess_teams = [i for i in range(3) if excess[i] > 0]
        deficit_teams = [i for i in range(3) if excess[i] < 0]
        
        # If there are no excess or deficit teams, we are done
        if not excess_teams or not deficit_teams:
            break
        
        # Try to balance the teams
        for e in excess_teams:
            for d in deficit_teams:
                # The amount we can transfer
                transfer_amount = min(excess[e], -excess[d])
                
                # We need to switch people from team e to team d
                # Count how many switches are needed
                count = 0
                for i in range(N):
                    if teams_strength[i][0] == e + 1 and count < transfer_amount:
                        count += 1
                        switches += 1
                    if count >= transfer_amount:
                        break
                
                # Update the excess values
                excess[e] -= transfer_amount
                excess[d] += transfer_amount
                
                # If we have balanced this team, we can break
                if excess[e] == 0:
                    break
            if excess[e] == 0:
                break
    
    # If we still have excess or deficit, it means we couldn't balance
    if any(e != 0 for e in excess):
        return -1
    
    return switches

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    teams_strength = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    result = min_switches_to_equal_strength(N, teams_strength)
    print(result)

if __name__ == "__main__":
    main()