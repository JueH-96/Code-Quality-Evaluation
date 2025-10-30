# YOUR CODE HERE
n, m = map(int, input().split())
hands = list(map(int, input().split()))

remaining_disinfectant = m
count = 0

for h in hands:
    if remaining_disinfectant >= h:
        remaining_disinfectant -= h
        count += 1
    else:
        remaining_disinfectant = 0

print(count)