S = input().strip()
correct_before = 0
incorrect_before = 0
result = []
for c in S:
    pos = incorrect_before + correct_before + 1
    result.append(str(pos))
    correct_before += 1
print(' '.join(result))