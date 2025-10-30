from collections import defaultdict

def perform_operations(length, colors, string, color_indices):
    for color in range(1, colors + 1):
        color_chars = []
        next_char_index = 0
        indices_of_color = color_indices[color]
        
        # Create circular list of indices
        indices_of_color.append(indices_of_color[0])
        
        for i in range(len(indices_of_color) - 1):
            if next_char_index != indices_of_color[i]:
                color_chars.append(string[next_char_index])
                next_char_index += 1
            else:
                # Once a cycle is found, add the rest of the string up to the start of the next cycle
                color_chars += list(string[next_char_index: indices_of_color[i]])
                next_char_index = indices_of_color[i + 1]
        
        # Perform circular shift
        shifted_chars = color_chars[-1:] + color_chars[:-1]
        
        # Replace the original color indices with the shifted characters
        for i, index in enumerate(indices_of_color[:-1]):
            string = string[:index] + shifted_chars[i] + string[index + 1:]
    
    return string

# Parsing the input
length, colors = map(int, input().split())
string = input()
color_indices = defaultdict(list)
color_data = list(map(int, input().split()))

# Filling the color indices dictionary
for index, color in enumerate(color_data):
    color_indices[color].append(index)

# Compute and print the result
result = perform_operations(length, colors, string, color_indices)
print(result)