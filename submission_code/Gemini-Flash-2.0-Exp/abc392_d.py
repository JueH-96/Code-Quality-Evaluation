def solve():
    n = int(input())
    dice = []
    for _ in range(n):
        line = list(map(int, input().split()))
        k = line[0]
        faces = line[1:]
        dice.append(faces)

    max_prob = 0
    for i in range(n):
        for j in range(i + 1, n):
            faces1 = dice[i]
            faces2 = dice[j]
            k1 = len(faces1)
            k2 = len(faces2)
            
            common_faces = set(faces1) & set(faces2)
            
            prob = 0
            for face in common_faces:
                count1 = faces1.count(face)
                count2 = faces2.count(face)
                prob += (count1 / k1) * (count2 / k2)
            
            max_prob = max(max_prob, prob)

    print(max_prob)

solve()