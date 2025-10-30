# YOUR CODE HERE
import sys

def main():
    """
    Reads a keyboard layout and calculates the total finger distance
    to type the alphabet in sequential order.
    """
    # Read the keyboard layout from a single line of standard input.
    try:
        S = sys.stdin.readline().strip()
    except (IOError, IndexError):
        # Handle potential empty input in a robust way, though
        # problem constraints likely guarantee valid input.
        return

    # Create a dictionary to map each character to its 0-indexed position.
    # This provides efficient O(1) average time complexity for lookups.
    # e.g., for S="QWERTY...", pos_map will be {'Q': 0, 'W': 1, 'E': 2, ...}
    pos_map = {char: i for i, char in enumerate(S)}

    # The required typing sequence is the standard English alphabet.
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Initialize total traveled distance.
    total_distance = 0

    # The finger starts at 'A'. We calculate the distance for each subsequent move.
    # The loop iterates 25 times for the moves: A->B, B->C, ..., Y->Z.
    for i in range(len(alphabet) - 1):
        # Get the current character and the next one in the typing sequence.
        current_char = alphabet[i]
        next_char = alphabet[i+1]

        # Find their positions on the keyboard using the map.
        pos_current = pos_map[current_char]
        pos_next = pos_map[next_char]

        # The distance for one move is the absolute difference of the positions.
        # The problem's 1-based coordinates yield the same distance difference
        # as 0-based indices: abs((pos_next + 1) - (pos_current + 1)) == abs(pos_next - pos_current).
        distance_for_move = abs(pos_next - pos_current)

        # Add this move's distance to the total.
        total_distance += distance_for_move

    # Print the final calculated total distance to standard output.
    print(total_distance)


if __name__ == "__main__":
    main()