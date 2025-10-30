from collections import Counter

n = int(input())
dice = []

for _ in range(n):
    line = list(map(int, input().split()))
    k = line[0]
    faces = line[1:]
    dice.append(faces)

max_prob = 0.0

# Try all pairs of dice
for i in range(n):
    for j in range(i + 1, n):
        dice1 = dice[i]
        dice2 = dice[j]
        
        # Count frequency of each face value for both dice
        count1 = Counter(dice1)
        count2 = Counter(dice2)
        
        # Calculate probability of getting the same number
        prob = 0.0
        
        # For each number that appears on both dice
        for num in count1:
            if num in count2:
                # Probability = (freq1/total1) * (freq2/total2)
                prob += (count1[num] / len(dice1)) * (count2[num] / len(dice2))
        
        max_prob = max(max_prob, prob)

print(max_prob)