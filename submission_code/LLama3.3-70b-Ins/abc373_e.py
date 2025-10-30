import sys

def solve(N, M, K, A):
    """
    Calculate the minimum number of additional votes each candidate needs to guarantee their victory.

    Args:
    N (int): The number of candidates.
    M (int): The maximum number of candidates that can have more votes than the current candidate.
    K (int): The total number of votes.
    A (list): A list of votes each candidate has received so far.

    Returns:
    list: A list of minimum additional votes each candidate needs to guarantee their victory.
    """
    # Calculate the total votes counted so far
    total_votes_counted = sum(A)

    # Calculate the remaining votes
    remaining_votes = K - total_votes_counted

    # Initialize the result list with -1
    result = [-1] * N

    # Iterate over each candidate
    for i in range(N):
        # Initialize the minimum additional votes needed to 0
        min_additional_votes = 0

        # Iterate over possible additional votes
        for additional_votes in range(remaining_votes + 1):
            # Initialize the count of candidates with more votes to 0
            count = 0

            # Iterate over each candidate
            for j in range(N):
                # If the current candidate is not the same as the candidate we are checking
                if i != j:
                    # If the current candidate has more votes than the candidate we are checking
                    if A[j] > A[i] + additional_votes:
                        # Increment the count
                        count += 1

            # If the count is less than M, it means the candidate can guarantee their victory
            if count < M:
                # Update the minimum additional votes needed
                min_additional_votes = additional_votes
                break

        # If the minimum additional votes needed is still 0, it means the candidate cannot guarantee their victory
        if min_additional_votes == 0 and i != 0:
            continue

        # Update the result list
        result[i] = min_additional_votes

    return result

def main():
    # Read the input from stdin
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the minimum number of additional votes each candidate needs
    result = solve(N, M, K, A)

    # Print the result to stdout
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()