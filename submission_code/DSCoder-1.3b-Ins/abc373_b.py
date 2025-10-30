# YOUR CODE HERE

def solve(s):
    keys = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    key_to_index = {key: index for index, key in enumerate(keys)}
    total_distance = 0
    current_index = key_to_index[s[0]]
    for key in s[1:]:
        next_index = key_to_index[key]
        distance = abs(current_index - next_index)
        total_distance += distance
        current_index = next_index
    return total_distance

s = input().strip()
print(solve(s))