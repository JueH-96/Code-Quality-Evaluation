# YOUR CODE HERE
import sys

def solve():
    # Read the number of people, N
    n = int(sys.stdin.readline())
    # Read the programming abilities P_1, P_2, ..., P_N as a list of integers
    p = list(map(int, sys.stdin.readline().split()))

    # Handle the edge case where there is only one person.
    # If N = 1, person 1 is trivially the strongest, needing 0 additional points.
    if n == 1:
        print(0)
        return

    # Get the score of person 1 (P_1). We use 0-based indexing, so it's p[0].
    p1_score = p[0]
    
    # Find the maximum score among all other people (persons 2 through N).
    # The slice p[1:] creates a list containing the scores P_2, P_3, ..., P_N.
    # We use the built-in max() function to find the largest score in this list.
    # Since we handled the n=1 case, n must be >= 2 here, ensuring p[1:] is not empty.
    max_other_score = max(p[1:]) 

    # To be the strongest, person 1's score must be strictly greater than the maximum score of any other person.
    # The minimum integer score required for person 1 to achieve this is max_other_score + 1.
    target_score = max_other_score + 1
    
    # Calculate the minimum additional points 'x' needed for person 1.
    # This is the difference between the target score required and person 1's current score.
    # x = target_score - p1_score
    needed_increase = target_score - p1_score
    
    # The problem states that x must be a non-negative integer.
    # If person 1's score is already greater than max_other_score (i.e., p1_score > max_other_score),
    # then target_score = max_other_score + 1 <= p1_score, so needed_increase will be <= 0.
    # In this scenario, person 1 doesn't need any more points, so the answer should be 0.
    # The max(0, needed_increase) function correctly handles this, ensuring the result is always non-negative.
    result = max(0, needed_increase)
    
    # Print the final answer, which is the minimum non-negative integer x.
    print(result)

# Execute the solving function to read input, process, and print output.
solve()