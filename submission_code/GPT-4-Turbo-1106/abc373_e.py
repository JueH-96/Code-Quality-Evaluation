import sys

def main():
    # Read input
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the remaining votes
    remaining_votes = K - sum(A)

    # Sort the candidates by the number of votes they have received
    sorted_candidates = sorted([(votes, idx) for idx, votes in enumerate(A)], reverse=True)

    # Initialize the result array with -1 (assuming no candidate can win)
    result = [-1] * N

    # The number of votes needed to be in the top M-1 candidates
    votes_needed = [0] * N
    if M > 1:
        for i in range(M - 1):
            votes_needed[sorted_candidates[i][1]] = sorted_candidates[M - 1][0] - sorted_candidates[i][0] + 1

    # Calculate the minimum votes needed for each candidate
    for i in range(N):
        # If the candidate is already in the top M-1, they don't need any additional votes
        if i < M - 1:
            result[sorted_candidates[i][1]] = max(0, votes_needed[sorted_candidates[i][1]])
        else:
            # Calculate the votes needed to surpass the M-1th candidate
            votes_to_win = sorted_candidates[M - 1][0] - sorted_candidates[i][0] + 1
            # If the candidate can get enough votes from the remaining votes, update the result
            if votes_to_win <= remaining_votes:
                result[sorted_candidates[i][1]] = votes_to_win

    # Print the result
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()