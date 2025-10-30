def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        a, b = map(int, input().split())
        chords.append(sorted((a, b)))

    for i in range(n):
        for j in range(i + 1, n):
            a1, b1 = chords[i]
            a2, b2 = chords[j]

            if (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1):
                print("Yes")
                return
    
    print("No")

solve()