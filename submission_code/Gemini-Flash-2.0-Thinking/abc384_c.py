from itertools import combinations

scores = list(map(int, input().split()))
problem_scores = {'A': scores[0], 'B': scores[1], 'C': scores[2], 'D': scores[3], 'E': scores[4]}
problems = ['A', 'B', 'C', 'D', 'E']
participants = []

for i in range(1, 6):
    for combo_tuple in combinations(problems, i):
        participant_name = "".join(sorted(combo_tuple))
        score = sum(problem_scores[problem] for problem in combo_tuple)
        participants.append((participant_name, score))

participants.sort(key=lambda item: (-item[1], item[0]))

for participant in participants:
    print(participant[0])