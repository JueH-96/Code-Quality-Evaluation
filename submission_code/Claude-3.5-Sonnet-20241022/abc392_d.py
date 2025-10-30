def calculate_probability(dice1, dice2):
    # Calculate probability of getting same number from two dice
    total = 0
    for num in set(dice1) & set(dice2):  # Common numbers between dice
        # Probability = sum of (probability of num on dice1 * probability of num on dice2)
        prob1 = dice1.count(num) / len(dice1)
        prob2 = dice2.count(num) / len(dice2)
        total += prob1 * prob2
    return total

# Read input
N = int(input())
dice = []
for _ in range(N):
    nums = list(map(int, input().split()))
    K = nums[0]  # Number of faces
    faces = nums[1:]  # Numbers on faces
    dice.append(faces)

# Calculate maximum probability by trying all pairs of dice
max_prob = 0
for i in range(N):
    for j in range(i + 1, N):
        prob = calculate_probability(dice[i], dice[j])
        max_prob = max(max_prob, prob)

# Print result
print(max_prob)