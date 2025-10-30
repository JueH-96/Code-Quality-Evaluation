point_to_index = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6
}
distances = [3, 1, 4, 1, 5, 9]

input_line = input().split()
p_str = input_line[0]
q_str = input_line[1]

p_index = point_to_index[p_str]
q_index = point_to_index[q_str]

start_index = min(p_index, q_index)
end_index = max(p_index, q_index)

total_distance = 0
for i in range(start_index, end_index):
    total_distance += distances[i]

print(total_distance)