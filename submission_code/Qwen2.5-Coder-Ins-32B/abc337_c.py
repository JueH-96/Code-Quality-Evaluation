# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))

    # Create a list to hold the positions
    positions = [-1] * (N + 1)

    # Find the person at the front of the line
    front_person = A.index(-1) + 1

    # Place the front person in the first position
    positions[0] = front_person

    # Fill in the rest of the positions
    current_index = 0
    while current_index < N:
        current_person = positions[current_index]
        next_person = A[current_person - 1]
        if next_person != -1:
            current_index += 1
            positions[current_index] = next_person

    # Print the result
    print(" ".join(map(str, positions[:N])))

if __name__ == "__main__":
    main()