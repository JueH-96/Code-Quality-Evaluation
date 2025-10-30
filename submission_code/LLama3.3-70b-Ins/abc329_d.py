import sys

def find_winner(votes, n):
    """
    Find the winner after counting the first i votes.

    Args:
    votes (list): A list of votes where each vote is for a candidate.
    n (int): The number of candidates.

    Returns:
    list: A list of winners after counting the first i votes.
    """
    winners = []
    vote_counts = [0] * (n + 1)

    for i in range(len(votes)):
        vote_counts[votes[i]] += 1
        max_votes = max(vote_counts[1:])
        winners.append(vote_counts.index(max_votes))

    return winners

def main():
    """
    Read the input from stdin and write the answer to stdout.
    """
    n, m = map(int, sys.stdin.readline().split())
    votes = list(map(int, sys.stdin.readline().split()))

    winners = find_winner(votes, n)

    for winner in winners:
        print(winner)

if __name__ == "__main__":
    main()