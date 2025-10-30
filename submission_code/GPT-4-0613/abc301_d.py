import itertools

S = input().strip()
N = int(input().strip())

question_marks = S.count('?')
combinations = list(itertools.product(['0', '1'], repeat=question_marks))

max_value = -1
for combination in combinations:
    temp_S = S
    for c in combination:
        temp_S = temp_S.replace('?', c, 1)
    value = int(temp_S, 2)
    if value <= N:
        max_value = max(max_value, value)

print(max_value)