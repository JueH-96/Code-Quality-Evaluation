# YOUR CODE HERE
def solve():
    n = int(input())
    chords = []
    for _ in range(n):
        chords.append(list(map(int, input().split())))

    def intersect(c1, c2):
        a1, b1 = c1
        a2, b2 = c2
        
        if a1 > b1:
            a1, b1 = b1, a1
        if a2 > b2:
            a2, b2 = b2, a2

        return (a1 < a2 < b1 < b2) or (a2 < a1 < b2 < b1)

    for i in range(n):
        for j in range(i + 1, n):
            if intersect(chords[i], chords[j]):
                print("Yes")
                return
    print("No")

solve()