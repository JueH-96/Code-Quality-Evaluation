# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))
    
    # We need to find the minimum score for the N-th round
    # Let's sort the scores of the first N-1 rounds
    A.sort()
    
    # We will try to find the minimum score for the N-th round
    # that makes the sum of the middle N-2 scores at least X
    # We need to add a new score to A and sort again
    # Then we will sum the middle N-2 scores and check if it's >= X
    
    # Try scores from 0 to 100 for the N-th round
    for score in range(101):
        # Create a new list with the N-th score included
        new_scores = A + [score]
        new_scores.sort()
        
        # Calculate the sum of the middle N-2 scores
        final_grade = sum(new_scores[1:N-1])
        
        # Check if this final grade is at least X
        if final_grade >= X:
            print(score)
            return
    
    # If no score from 0 to 100 can achieve the required final grade
    print(-1)

if __name__ == "__main__":
    main()