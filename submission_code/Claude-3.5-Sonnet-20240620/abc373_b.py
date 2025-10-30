# YOUR CODE HERE
def calculate_distance(keyboard):
    positions = {char: i+1 for i, char in enumerate(keyboard)}
    current_pos = positions['A']
    total_distance = 0
    
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        target_pos = positions[char]
        distance = abs(target_pos - current_pos)
        total_distance += distance
        current_pos = target_pos
    
    return total_distance

keyboard = input().strip()
print(calculate_distance(keyboard))