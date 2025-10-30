N, M = map(int, input().split())
photos = [list(map(int, input().split())) for _ in range(M)]

bad_mood = [[True]*N for _ in range(N)]
for photo in photos:
    for i in range(N-1):
        bad_mood[photo[i]-1][photo[i+1]-1] = False
        bad_mood[photo[i+1]-1][photo[i]-1] = False

count = 0
for i in range(N):
    for j in range(i+1, N):
        if bad_mood[i][j]:
            count += 1

print(count)