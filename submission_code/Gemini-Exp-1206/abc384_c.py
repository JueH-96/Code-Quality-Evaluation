def solve():
    scores = list(map(int, input().split()))
    participants = []
    for i in range(1, 32):
        name = ""
        score = 0
        for j in range(5):
            if (i >> j) & 1:
                name += chr(ord('A') + j)
                score += scores[j]
        participants.append((score, name))
    
    participants.sort(key=lambda x: (-x[0], x[1]))
    
    for score, name in participants:
        print(name)

solve()