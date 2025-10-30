# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Step 1: Find the overall largest integer in the list.
    # Python's built-in max() function is efficient for this.
    max_val = max(A)

    # Step 2: Find the largest integer among those that are not max_val.
    # Initialize 'second_max' with a value smaller than any possible A_i (1 <= A_i <= 100).
    # -1 is a safe choice.
    second_max = -1 

    for x in A:
        # If the current number x is not the overall maximum,
        # it is a candidate for 'second_max'.
        if x != max_val:
            # Update 'second_max' if x is larger than the current 'second_max'.
            if x > second_max:
                second_max = x
    
    # Print the result.
    print(second_max)

solve()