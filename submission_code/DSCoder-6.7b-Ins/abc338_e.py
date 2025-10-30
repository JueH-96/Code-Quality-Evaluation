def check_intersection(chords):
    chords.sort()
    for i in range(len(chords) - 1):
        if chords[i][1] > chords[i + 1][0]:
            return "Yes"
    return "No"

N = int(input())
chords = [list(map(int, input().split())) for _ in range(N)]
print(check_intersection(chords))