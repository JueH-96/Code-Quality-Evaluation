def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        a, b = map(int, input().split())
        chords.append((a, b))

    def intersect(chord1, chord2):
        a1, b1 = sorted(chord1)
        a2, b2 = sorted(chord2)
        return (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1)

    for i in range(n):
        for j in range(i + 1, n):
            if intersect(chords[i], chords[j]):
                print("Yes")
                return

    print("No")

solve()