def solve():
    N, X = map(int, input().split())
    A = list(map(int, input().split())) # A has N-1 elements

    # Iterate through all possible scores for the Nth round (0 to 100)
    for score_N_candidate in range(101):
        # Create the full list of N scores by adding the candidate score for round N
        # A is a list of N-1 scores. Adding one more makes it N scores.
        all_scores = A + [score_N_candidate]
        
        # Sort the scores in ascending order
        all_scores.sort()
        
        # Calculate the final grade.
        # The problem states: "let S=(S_1,S_2,...,S_N) be the sequence of the scores
        # earned in the rounds sorted in ascending order, then the final grade is
        # S_2+S_3+...+S_{N-1}."
        # In a 0-indexed Python list `all_scores` of length N:
        # S_1 is all_scores[0]
        # S_2 is all_scores[1]
        # ...
        # S_{N-1} is all_scores[N-2]
        # S_N is all_scores[N-1]
        # So the sum is all_scores[1] + all_scores[2] + ... + all_scores[N-2].
        # This sum can be obtained by slicing: sum(all_scores[1:N-1]).
        # Note: The slice all_scores[1:N-1] includes elements from index 1 up to (N-1)-1 = N-2.
        # The number of elements summed is (N-2) - 1 + 1 = N-2.
        # This is correct as per problem statement ("sum of the N-2 of the scores").
        # Constraints: N >= 3.
        # If N=3, slice is all_scores[1:2], which sums just all_scores[1] (i.e., S_2). Correct.
        
        calculated_grade = sum(all_scores[1:N-1])
        
        # Check if this grade meets the target X
        if calculated_grade >= X:
            print(score_N_candidate)
            return
            
    # If the loop completes, it means no score from 0 to 100 (inclusive)
    # for the Nth round can achieve a final grade of X or higher.
    print(-1)

if __name__ == '__main__':
    solve()