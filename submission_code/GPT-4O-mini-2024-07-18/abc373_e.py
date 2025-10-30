def minimum_additional_votes(N, M, K, A):
    total_votes = sum(A)
    remaining_votes = K - total_votes
    results = []

    # Sort the current votes to determine the ranking
    sorted_votes = sorted(A, reverse=True)

    for i in range(N):
        current_votes = A[i]
        # Determine how many candidates have more votes than candidate i
        count_higher = sum(1 for v in sorted_votes if v > current_votes)

        # If candidate i is already elected
        if count_higher < M:
            results.append(0)
            continue
        
        # Calculate the minimum votes needed to ensure victory
        needed_votes = count_higher - (M - 1)
        if needed_votes < 0:
            results.append(0)
            continue
        
        # Calculate the maximum votes that can be given to other candidates
        max_other_votes = remaining_votes
        
        # If the needed votes exceed the available votes, it's impossible
        if needed_votes > max_other_votes:
            results.append(-1)
        else:
            results.append(needed_votes)

    return results

import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3 + N]))

    results = minimum_additional_votes(N, M, K, A)
    print(" ".join(map(str, results)))

if __name__ == "__main__":
    main()