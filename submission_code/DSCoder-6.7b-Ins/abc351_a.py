# YOUR CODE HERE

def min_runs_to_win():
    # Read the inputs
    takahashi_scores = list(map(int, input().split()))
    aoki_scores = list(map(int, input().split()))

    # Calculate the total scores
    takahashi_total = sum(takahashi_scores)
    aoki_total = sum(aoki_scores)

    # Calculate the minimum runs needed to win
    min_runs = aoki_total - takahashi_total + 1

    # Print the result
    print(min_runs)

min_runs_to_win()