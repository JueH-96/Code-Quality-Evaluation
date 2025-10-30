import collections

def solve():
    n = int(input())
    people = []
    for _ in range(n):
        line = list(map(int, input().split()))
        people.append({'team': line[0], 'strength': line[1], 'index': _})
    
    initial_assignment = tuple(person['team'] for person in people)
    
    initial_team_strengths = [0, 0, 0]
    for person in people:
        initial_team_strengths[person['team'] - 1] += person['strength']
        
    total_strength = sum(initial_team_strengths)
    if total_strength % 3 != 0:
        print("-1")
        return
        
    target_strength = total_strength // 3
    
    if initial_team_strengths[0] == target_strength and initial_team_strengths[1] == target_strength and initial_team_strengths[2] == target_strength:
        print(0)
        return
        
    queue = collections.deque([(initial_assignment, 0)])
    visited_assignments = {initial_assignment}
    
    while queue:
        current_assignment, moves = queue.popleft()
        
        current_team_strengths = [0, 0, 0]
        for i in range(n):
            current_team_strengths[current_assignment[i] - 1] += people[i]['strength']
            
        if current_team_strengths[0] == target_strength and current_team_strengths[1] == target_strength and current_team_strengths[2] == target_strength:
            print(moves)
            return
            
        for i in range(n):
            original_team = current_assignment[i]
            for next_team in [1, 2, 3]:
                if next_team != original_team:
                    next_assignment_list = list(current_assignment)
                    next_assignment_list[i] = next_team
                    next_assignment = tuple(next_assignment_list)
                    if next_assignment not in visited_assignments:
                        visited_assignments.add(next_assignment)
                        queue.append((next_assignment, moves + 1))
                        
    print("-1") # Should not reach here if it's always possible when total strength is divisible by 3

if __name__ == '__main__':
    solve()