# YOUR CODE HERE
a,b,c,d,e = map(int, input().split())
problems = [('A',a), ('B',b), ('C',c), ('D',d), ('E',e)]
participants = []
for num in range(1,32):
    combination = []
    total_score = 0
    for pos in range(5):
        if (num >> (4 - pos)) & 1:
            problem_name, score = problems[pos]
            combination.append(problem_name)
            total_score += score
    participant_name = ''.join(combination)
    participants.append((-total_score, participant_name))
participants.sort()
for p in participants:
    print(p[1])