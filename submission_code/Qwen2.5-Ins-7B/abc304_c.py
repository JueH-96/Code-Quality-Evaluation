# YOUR CODE HERE
n, d = map(int, input().split())
people = [tuple(map(int, input().split())) for _ in range(n)]
infected = [False] * n
infected[0] = True

def is_within_distance(p1, p2, d):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 <= d

for i in range(n):
    if not infected[i]:
        continue
    for j in range(n):
        if i == j:
            continue
        if is_within_distance(people[i], people[j], d):
            infected[j] = True

for i in range(n):
    print('Yes' if infected[i] else 'No')