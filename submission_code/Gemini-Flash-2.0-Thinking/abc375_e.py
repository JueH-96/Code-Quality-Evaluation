from collections import deque

def solve():
    n = int(input())
    people = []
    for _ in range(n):
        a, b = map(int, input().split())
        people.append((a, b))

    total_strength = sum(b for _, b in people)
    if total_strength % 3 != 0:
        print(-1)
        return

    target_strength = total_strength // 3

    initial_assignment = [p[0] for p in people]

    def calculate_strengths(assignment):
        strengths = [0, 0, 0]
        for i in range(n):
            strengths[assignment[i] - 1] += people[i][1]
        return strengths

    initial_strengths = calculate_strengths(initial_assignment)
    if all(s == target_strength for s in initial_strengths):
        print(0)
        return

    for num_switches in range(1, n + 1):
        queue = deque([(list(initial_assignment), 0)])
        visited = {tuple(initial_assignment)}

        while queue:
            current_assignment, current_switches = queue.popleft()

            current_strengths = calculate_strengths(current_assignment)
            if all(s == target_strength for s in current_strengths):
                print(num_switches)
                return

            if current_switches < num_switches:
                for i in range(n):
                    person_id = i + 1
                    current_team = current_assignment[i]
                    for next_team in [1, 2, 3]:
                        if next_team != current_team:
                            next_assignment = list(current_assignment)
                            next_assignment[i] = next_team
                            if tuple(next_assignment) not in visited:
                                visited.add(tuple(next_assignment))
                                queue.append((next_assignment, current_switches + 1))

    print(-1)

solve()