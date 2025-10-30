def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))
    
    # We need to find the minimum score in the last round to make the sum of middle N-2 scores >= X
    # First, sort the existing scores
    A.sort()
    
    # Calculate the sum of the middle N-2 scores if we assume the last score is 0
    # This is the worst case for the last score, as it could be the new minimum
    if N - 1 > 1:
        current_sum = sum(A[1:N-1])  # Sum excluding the first and last of the sorted first N-1 elements
    else:
        current_sum = 0  # If N-1 is 1, then there's no middle element to sum
    
    # We need at least X in the sum of the middle N-2 scores
    required_sum = X
    
    # Calculate the minimum score needed in the last round to reach the required sum
    min_score_needed = required_sum - current_sum
    
    # If the minimum score needed is less than 0, we can achieve the required sum with a score of 0
    if min_score_needed <= 0:
        print(0)
        return
    
    # If the minimum score needed is more than 100, it's impossible to achieve
    if min_score_needed > 100:
        print(-1)
        return
    
    # Otherwise, the minimum score needed is the answer, if it's between 0 and 100
    print(min_score_needed)

if __name__ == "__main__":
    main()