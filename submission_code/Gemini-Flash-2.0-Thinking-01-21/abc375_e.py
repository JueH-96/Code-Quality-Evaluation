import collections

def solve():
    n = int(input())
    people = []
    initial_teams = []
    strengths = []
    for _ in range(n):
        a, b = map(int, input().split())
        initial_teams.append(a)
        strengths.append(b)
        people.append({'initial_team': a, 'strength': b, 'id': _ + 1})
    
    total_strength = sum(strengths)
    if total_strength % 3 != 0:
        print("-1")
        return
    
    target_strength = total_strength // 3
    
    initial_assignment = tuple(initial_teams)
    
    if calculate_team_strengths(initial_assignment, strengths) == [target_strength, target_strength, target_strength]:
        print(0)
        return
        
    queue = collections.deque([(initial_assignment, 0)])
    visited_states = {initial_assignment}
    
    while queue:
        current_assignment, moves_count = queue.popleft()
        
        for person_index in range(n):
            current_team = current_assignment[person_index]
            for next_team in [1, 2, 3]:
                if next_team != current_team:
                    next_assignment_list = list(current_assignment)
                    next_assignment_list[person_index] = next_team
                    next_assignment = tuple(next_assignment_list)
                    if next_assignment not in visited_states:
                        visited_states.add(next_assignment)
                        team_strengths = calculate_team_strengths(next_assignment, strengths)
                        if team_strengths == [target_strength, target_strength, target_strength]:
                            print(moves_count + 1)
                            return
                        queue.append((next_assignment, moves_count + 1))
                        
    print("-1")

def calculate_team_strengths(assignment, strengths):
    team_strengths = [0, 0, 0]
    for i in range(len(assignment)):
        team_index = assignment[i] - 1
        team_strengths[team_index] += strengths[i]
    return team_strengths

if __name__ == '__main__':
    solve()