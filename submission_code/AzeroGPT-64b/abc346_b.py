W, B = map(int, input().split())

# S is infinitely repeating wbwbwwbwbwbw
# So, we just need to check with a total of 12 key repeats.
WINDOW = 'wbwbwwbwbwbw'

for window_size in range(12):
    pattern = WINDOW[:window_size]

    if pattern.count('w') >= W and pattern.count('b') >= B or pattern.count('b') >= W and pattern.count('w') >= B:
        print('Yes')
        break
else:
    print('No')