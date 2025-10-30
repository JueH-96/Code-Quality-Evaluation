# YOUR CODE HERE
import sys

def main():
    # Read the number of matches from the first line of standard input
    n = int(sys.stdin.readline())

    # Initialize total scores for Team Takahashi and Team Aoki to zero
    total_takahashi = 0
    total_aoki = 0

    # Loop N times, once for each match
    for _ in range(n):
        # Read the scores for the current match from standard input.
        # sys.stdin.readline() reads a line (e.g., "10 2").
        # .split() splits the line into a list of strings based on whitespace (e.g., ["10", "2"]).
        # map(int, ...) applies the int function to each element of the list,
        # converting the strings to integers (e.g., [10, 2]).
        # The resulting integers are unpacked into variables x (Takahashi's score) and y (Aoki's score).
        x, y = map(int, sys.stdin.readline().split())
        
        # Add the scores obtained in this match to the respective team's total score accumulator
        total_takahashi += x
        total_aoki += y

    # After iterating through all the matches and summing the scores,
    # compare the final total scores to determine the winner or if it's a draw.
    if total_takahashi > total_aoki:
        # If Team Takahashi's total score is strictly greater than Team Aoki's, Team Takahashi wins.
        print("Takahashi")
    elif total_aoki > total_takahashi:
        # If Team Aoki's total score is strictly greater than Team Takahashi's, Team Aoki wins.
        print("Aoki")
    else: # This condition is met if total_takahashi == total_aoki
        # If the total scores are equal, the result is a draw.
        print("Draw")

# This is a standard Python construct. It ensures that the main() function is called 
# only when the script is executed directly (not when it's imported as a module).
if __name__ == '__main__':
    main()
# END YOUR CODE HERE