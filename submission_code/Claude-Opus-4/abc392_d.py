# YOUR CODE HERE
from collections import Counter

N = int(input())
dice = []

for _ in range(N):
    line = list(map(int, input().split()))
    K = line[0]
    faces = line[1:K+1]
    dice.append((K, faces))

max_prob = 0.0

# Check all pairs of dice
for i in range(N):
    for j in range(i+1, N):
        K1, faces1 = dice[i]
        K2, faces2 = dice[j]
        
        # Count occurrences of each number on each die
        count1 = Counter(faces1)
        count2 = Counter(faces2)
        
        # Calculate probability of getting same number
        prob = 0.0
        
        # Find common numbers between the two dice
        common_numbers = set(faces1) & set(faces2)
        
        for num in common_numbers:
            # Probability of getting 'num' on die1 * probability of getting 'num' on die2
            prob += (count1[num] / K1) * (count2[num] / K2)
        
        max_prob = max(max_prob, prob)

print(f"{max_prob:.15f}")