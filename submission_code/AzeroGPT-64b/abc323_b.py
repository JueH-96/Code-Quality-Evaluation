n = int(input())
s = [input() for _ in range(n)]
win_counts = [(i, sum(1 for c in s[i] if c == 'o')) for i in range(n)]
win_counts.sort(key=lambda x: (-x[1], x[0]))
for i, _ in win_counts:
    print(i + 1)