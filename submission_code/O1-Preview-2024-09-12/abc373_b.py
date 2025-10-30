# YOUR CODE HERE
S = input()
positions = {c: i+1 for i, c in enumerate(S)}
positions_list = [positions[c] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
total_distance = sum(abs(positions_list[i] - positions_list[i-1]) for i in range(1, len(positions_list)))
print(total_distance)