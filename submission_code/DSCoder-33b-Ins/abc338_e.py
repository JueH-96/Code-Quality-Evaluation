# YOUR CODE HERE

N = int(input())
chords = []
for _ in range(N):
    A, B = map(int, input().split())
    chords.append((A, B))

chords.sort()

for i in range(N-1):
    if chords[i][1] > chords[i+1][0]:
        print("Yes")
        break
else:
    print("No")