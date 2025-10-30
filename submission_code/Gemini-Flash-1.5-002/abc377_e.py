n, k = map(int, input().split())
p = list(map(int, input().split()))

p = [x - 1 for x in p]  # Adjust to 0-based indexing

history = []
current_p = list(p)
for _ in range(min(k, n + 10)):  # Iterate at most n+10 times to find cycle
    next_p = [current_p[x] for x in current_p]
    if next_p in history:
        first_occurrence = history.index(next_p)
        remaining_iterations = (k - len(history)) % (len(history) - first_occurrence)
        current_p = history[first_occurrence + remaining_iterations]
        break
    history.append(list(current_p))
    current_p = next_p

print(*[x + 1 for x in current_p])