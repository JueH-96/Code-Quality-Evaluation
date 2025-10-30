# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:]))
    
    sum_A = sum(A)
    remaining_votes = K - sum_A
    
    # Sort the votes to determine the threshold for being in the top M
    sorted_A = sorted(A)
    
    # Result list
    results = []
    
    for i in range(N):
        current_votes = A[i]
        
        # Determine the threshold to be in the top M
        # We need to be better than at least N-M candidates
        if M == N:
            # If M == N, everyone can be elected, so no additional votes needed
            results.append(0)
            continue
        
        # The threshold is the M-th highest vote count
        threshold_index = N - M
        if threshold_index > 0:
            threshold_votes = sorted_A[threshold_index - 1]
        else:
            threshold_votes = -1
        
        # Calculate the minimum additional votes needed
        if current_votes > threshold_votes:
            # Already in the top M
            results.append(0)
        else:
            # Need to surpass the threshold_votes
            needed_votes = threshold_votes - current_votes + 1
            if needed_votes <= remaining_votes:
                results.append(needed_votes)
            else:
                results.append(-1)
    
    print(' '.join(map(str, results)))