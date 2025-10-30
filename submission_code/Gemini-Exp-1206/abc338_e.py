def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        chords.append(list(map(int, input().split())))

    def intersect(c1, c2):
        a, b = sorted(c1)
        c, d = sorted(c2)
        if a < c < b < d or c < a < d < b:
            return True
        return False

    for i in range(n):
        for j in range(i + 1, n):
            if intersect(chords[i], chords[j]):
                print("Yes")
                return
    print("No")

solve()