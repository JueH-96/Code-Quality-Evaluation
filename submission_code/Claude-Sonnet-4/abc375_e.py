from collections import deque

def solve():
    n = int(input())
    people = []
    for i in range(n):
        a, b = map(int, input().split())
        people.append((a-1, b))  # Convert to 0-indexed teams
    
    # Calculate total strength
    total_strength = sum(b for _, b in people)
    
    # Check if divisible by 3
    if total_strength % 3 != 0:
        print(-1)
        return
    
    target = total_strength // 3
    
    # Calculate initial team strengths
    initial_strengths = [0, 0, 0]
    for team, strength in people:
        initial_strengths[team] += strength
    
    # If already balanced, return 0
    if all(s == target for s in initial_strengths):
        print(0)
        return
    
    # BFS with state compression
    # State: current assignment of people to teams
    initial_state = tuple(people[i][0] for i in range(n))
    queue = deque([(initial_state, 0)])
    visited = set([initial_state])
    
    while queue:
        state, switches = queue.popleft()
        
        # Calculate current team strengths
        strengths = [0, 0, 0]
        for i in range(n):
            strengths[state[i]] += people[i][1]
        
        # Check if balanced
        if all(s == target for s in strengths):
            print(switches)
            return
        
        # Try switching each person
        for i in range(n):
            current_team = state[i]
            for new_team in range(3):
                if new_team != current_team:
                    new_state = list(state)
                    new_state[i] = new_team
                    new_state_tuple = tuple(new_state)
                    
                    if new_state_tuple not in visited:
                        visited.add(new_state_tuple)
                        queue.append((new_state_tuple, switches + 1))
    
    print(-1)

solve()