# YOUR CODE HERE
import sys
import bisect

def main():
    N, M, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    total_votes = sum(A)
    remaining_votes = K - total_votes
    
    # Create a list of tuples (A_i, i) to keep track of original indices
    indexed_A = [(A[i], i) for i in range(N)]
    # Sort based on A_i in descending order
    indexed_A.sort(reverse=True, key=lambda x: x[0])
    
    # Extract sorted A and original indices
    sorted_A = [x[0] for x in indexed_A]
    original_indices = [x[1] for x in indexed_A]
    
    # Precompute the prefix sums of the sorted A
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i+1] = prefix_sum[i] + sorted_A[i]
    
    # Initialize the result array
    result = [0] * N
    
    for i in range(N):
        current_A = sorted_A[i]
        # To ensure that the candidate is in the top M, we need to make sure that at most M-1 candidates have more votes than current_A + X
        # So, we need to find the M-th candidate's votes and ensure that current_A + X >= that value
        
        # The M-th candidate's votes is sorted_A[M-1]
        # So, current_A + X >= sorted_A[M-1]
        # But we need to consider the case where current_A is already >= sorted_A[M-1]
        
        # Also, we need to consider the remaining votes
        
        # The minimum X is max(0, sorted_A[M-1] - current_A)
        # But we need to ensure that X <= remaining_votes
        
        # Also, if current_A + X > sorted_A[M-1], then it's possible to have more than M-1 candidates with more votes
        
        # So, to ensure that the candidate is in the top M, we need to make sure that the number of candidates with more than current_A + X is less than M
        
        # So, we need to find the smallest X such that the number of candidates with more than current_A + X is less than M
        
        # To find this, we can perform a binary search on X
        
        # The maximum possible X is remaining_votes
        
        # We need to find the smallest X such that the number of candidates with more than current_A + X is less than M
        
        # So, we can perform a binary search on X from 0 to remaining_votes
        
        # For each X, we can count the number of candidates with more than current_A + X
        
        # We can use bisect to find the number of candidates with more than current_A + X
        
        # The number of candidates with more than current_A + X is the number of candidates in sorted_A that are greater than current_A + X
        
        # So, we can use bisect_right to find the index where current_A + X would be inserted to maintain order
        
        # The number of candidates with more than current_A + X is N - bisect_right(sorted_A, current_A + X)
        
        # We need this to be less than M
        
        # So, N - bisect_right(sorted_A, current_A + X) < M
        
        # Which is equivalent to bisect_right(sorted_A, current_A + X) > N - M
        
        # So, we need to find the smallest X such that bisect_right(sorted_A, current_A + X) > N - M
        
        # We can perform a binary search on X to find the smallest X that satisfies this condition
        
        low = 0
        high = remaining_votes
        answer = -1
        
        while low <= high:
            mid = (low + high) // 2
            # Calculate the number of candidates with more than current_A + mid
            # Using bisect_right
            idx = bisect.bisect_right(sorted_A, current_A + mid)
            cnt = N - idx
            if cnt < M:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        if answer != -1:
            result[original_indices[i]] = answer
        else:
            result[original_indices[i]] = -1
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()