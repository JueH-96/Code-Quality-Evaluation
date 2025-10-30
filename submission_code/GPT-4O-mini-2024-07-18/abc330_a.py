# YOUR CODE HERE
def count_passed_students():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line for N and L
    N, L = map(int, data[0].split())
    
    # Read the second line for scores
    scores = list(map(int, data[1].split()))
    
    # Count how many scores are greater than or equal to L
    passed_count = sum(1 for score in scores if score >= L)
    
    # Print the result
    print(passed_count)

count_passed_students()