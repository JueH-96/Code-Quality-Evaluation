# YOUR CODE HERE
n = int(input())
dice = []

for _ in range(n):
    line = list(map(int, input().split()))
    k = line[0]
    faces = line[1:k+1]
    dice.append(faces)

max_prob = 0.0

for i in range(n):
    for j in range(i+1, n):
        # Calculate probability that dice i and j show the same number
        # Count occurrences of each number on each die
        count_i = {}
        count_j = {}
        
        for face in dice[i]:
            count_i[face] = count_i.get(face, 0) + 1
        
        for face in dice[j]:
            count_j[face] = count_j.get(face, 0) + 1
        
        # Calculate probability
        prob = 0.0
        for num in count_i:
            if num in count_j:
                prob_i = count_i[num] / len(dice[i])
                prob_j = count_j[num] / len(dice[j])
                prob += prob_i * prob_j
        
        max_prob = max(max_prob, prob)

print(max_prob)