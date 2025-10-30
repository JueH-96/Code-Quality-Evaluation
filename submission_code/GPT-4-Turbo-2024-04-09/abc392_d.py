import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    
    dice = []
    face_counts = [0] * 100001  # We can have face values from 1 to 100000
    
    for _ in range(N):
        K_i = int(data[index])
        index += 1
        faces = list(map(int, data[index:index+K_i]))
        index += K_i
        
        face_count = {}
        for face in faces:
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1
        
        dice.append((K_i, face_count))
        
        # Update global face counts
        for face, count in face_count.items():
            face_counts[face] += count
    
    max_probability = 0.0
    
    # Calculate the maximum probability of matching numbers
    for K_i, face_count_i in dice:
        for face, count_i in face_count_i.items():
            # Total occurrences of this face in all dice
            total_count = face_counts[face]
            # Remaining occurrences in other dice
            remaining_count = total_count - count_i
            if remaining_count > 0:
                # Probability of rolling 'face' on this die
                prob_i = count_i / K_i
                # Maximum probability with any other die
                max_probability = max(max_probability, prob_i * (remaining_count / (K_i * 100000)))
    
    print(f"{max_probability:.12f}")

main()