def max_probability_of_same_roll(N, dice):
    from collections import defaultdict
    
    max_prob = 0.0
    
    # Create a dictionary to count occurrences of each face value across all dice
    face_count = defaultdict(int)
    
    # Fill the face_count dictionary with the occurrences of each face
    for die in dice:
        K_i = die[0]
        faces = die[1:K_i + 1]
        for face in faces:
            face_count[face] += 1
    
    # Now we need to calculate the maximum probability for each pair of dice
    for i in range(N):
        K_i = dice[i][0]
        faces_i = dice[i][1:K_i + 1]
        
        for j in range(i + 1, N):
            K_j = dice[j][0]
            faces_j = dice[j][1:K_j + 1]
            
            # Count the number of common faces between die i and die j
            common_faces = set(faces_i) & set(faces_j)
            if common_faces:
                # Calculate the probability of rolling the same number
                prob = len(common_faces) / (K_i * K_j)
                max_prob = max(max_prob, prob)
    
    return max_prob

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
dice = []

for i in range(1, N + 1):
    line = list(map(int, data[i].split()))
    dice.append(line)

# Calculate the maximum probability
result = max_probability_of_same_roll(N, dice)

# Print the result with sufficient precision
print(f"{result:.12f}")