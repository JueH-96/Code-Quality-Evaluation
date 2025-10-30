import sys

def minimal_travel_distance(S):
    # Create a dictionary to map each character to its position
    position_map = {char: index for index, char in enumerate(S, start=1)}

    # Calculate the total travel distance
    total_distance = 0
    for i in range(1, 26):
        current_char = chr(ord('A') + i - 1)
        next_char = chr(ord('A') + i)
        total_distance += abs(position_map[next_char] - position_map[current_char])

    return total_distance

if __name__ == "__main__":
    input = sys.stdin.read().strip()
    result = minimal_travel_distance(input)
    print(result)