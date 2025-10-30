# YOUR CODE HERE
import sys

def main():
    S = sys.stdin.read().strip()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    current_position = S.index('A')
    total_distance = 0

    for char in alphabet[1:]:
        next_position = S.index(char)
        total_distance += abs(next_position - current_position)
        current_position = next_position

    print(total_distance)

if __name__ == "__main__":
    main()