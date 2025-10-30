N, K = map(int, input().split())
S = input()

current_run = 0
result = 0

for c in S:
    if c == 'O':
        current_run += 1
    else:
        result += current_run // K
        current_run = 0

result += current_run // K

print(result)