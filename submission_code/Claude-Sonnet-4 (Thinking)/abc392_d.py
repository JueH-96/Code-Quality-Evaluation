n = int(input())
dice = []

for i in range(n):
    line = list(map(int, input().split()))
    k = line[0]
    faces = line[1:]
    dice.append(faces)

max_prob = 0

for i in range(n):
    for j in range(i + 1, n):
        dice_i = dice[i]
        dice_j = dice[j]
        
        # Count occurrences of each number on each die
        count_i = {}
        count_j = {}
        
        for face in dice_i:
            count_i[face] = count_i.get(face, 0) + 1
        
        for face in dice_j:
            count_j[face] = count_j.get(face, 0) + 1
        
        # Calculate probability that both dice show the same number
        prob = 0
        for num in count_i:
            if num in count_j:
                prob += (count_i[num] / len(dice_i)) * (count_j[num] / len(dice_j))
        
        max_prob = max(max_prob, prob)

print(max_prob)