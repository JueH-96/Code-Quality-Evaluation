# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:2+N-1]))
    
    # We need to find the minimum score S_N such that the sum of the middle N-2 scores is >= X
    # The score S_N can be from 0 to 100
    
    # To find the minimum S_N, we can try all possible values from 0 to 100 and find the smallest one that satisfies the condition
    
    min_score = 101  # Initialize with a value larger than 100
    
    for s in range(0, 101):
        # Create the full list of scores
        scores = A.copy()
        scores.append(s)
        # Sort the scores
        scores_sorted = sorted(scores)
        # Calculate the sum of the middle N-2 scores
        sum_middle = sum(scores_sorted[1:-1])
        if sum_middle >= X:
            min_score = s
            break
    
    if min_score <= 100:
        print(min_score)
    else:
        print(-1)

if __name__ == "__main__":
    main()