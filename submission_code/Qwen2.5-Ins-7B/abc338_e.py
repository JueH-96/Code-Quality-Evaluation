# YOUR CODE HERE
n = int(input())
chords = []
for i in range(n):
    a, b = map(int, input().split())
    chords.append((a, b))

for i in range(n):
    for j in range(i + 1, n):
        a1, b1 = chords[i]
        a2, b2 = chords[j]
        if (min(a1, b1) < min(a2, b2) < max(a1, b1) < max(a2, b2)) or (min(a2, b2) < min(a1, b1) < max(a2, b2) < max(a1, b1)):
            print('Yes')
            exit()
print('No')