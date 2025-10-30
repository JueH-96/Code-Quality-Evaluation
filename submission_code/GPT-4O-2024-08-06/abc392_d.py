# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    
    dice = []
    
    for _ in range(N):
        K = int(data[index])
        index += 1
        faces = list(map(int, data[index:index + K]))
        index += K
        dice.append(faces)
    
    max_probability = 0.0
    
    for i in range(N):
        dice_i = dice[i]
        count_i = {}
        for face in dice_i:
            if face in count_i:
                count_i[face] += 1
            else:
                count_i[face] = 1
        K_i = len(dice_i)
        
        for j in range(i + 1, N):
            dice_j = dice[j]
            count_j = {}
            for face in dice_j:
                if face in count_j:
                    count_j[face] += 1
                else:
                    count_j[face] = 1
            K_j = len(dice_j)
            
            # Calculate the probability for this pair
            probability = 0.0
            for face in count_i:
                if face in count_j:
                    probability += (count_i[face] / K_i) * (count_j[face] / K_j)
            
            # Update the maximum probability
            if probability > max_probability:
                max_probability = probability
    
    # Print the maximum probability with sufficient precision
    print(f"{max_probability:.12f}")