# YOUR CODE HERE
def intersect(a, b, c, d):
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    return (a < c < b < d) or (c < a < d < b)

n = int(input())
chords = [tuple(map(int, input().split())) for _ in range(n)]

for i in range(n):
    for j in range(i + 1, n):
        if intersect(chords[i][0], chords[i][1], chords[j][0], chords[j][1]):
            print("Yes")
            exit()

print("No")