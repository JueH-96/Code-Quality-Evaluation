import sys

def main():
    # Read the number of players and the number of score changes
    N, T = map(int, sys.stdin.readline().split())

    # Initialize the scores of all players to zero
    scores = [0] * N

    # Process each score change and print the number of different scores
    for _ in range(T):
        A, B = map(int, sys.stdin.readline().split())
        scores[A - 1] += B  # Update the score of player A
        print(len(set(scores)))  # Print the number of different scores

if __name__ == "__main__":
    main()