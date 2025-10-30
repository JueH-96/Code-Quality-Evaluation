# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N (number of words) and M (maximum number of lines allowed)
    # N: total number of words in the sentence
    # M: maximum number of lines the sentence can occupy
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the list of word lengths L
    # L[i] is the width of the (i+1)-th word
    L = list(map(int, sys.stdin.readline().split()))

    # Define the check function: 
    # Determines if the sentence can be displayed within M lines given a maximum window width W.
    # Uses a greedy approach: fit as many words as possible onto each line.
    # Returns True if the sentence fits in M or fewer lines, False otherwise.
    def check(W):
        lines_needed = 1  # Start count with the first line
        current_line_width = 0 # Tracks the width used on the current line

        # Iterate through each word length in the list L
        for i in range(N):
            word_len = L[i]
            
            # Basic check: If a single word is longer than W, it's impossible to fit.
            # This check is technically redundant because the binary search guarantees W >= max(L).
            # If needed for clarity/robustness:
            # if word_len > W: 
            #    return False # A single word exceeds the window width

            if current_line_width == 0:
                # If the current line is empty (this word is the first on the line),
                # the line width becomes the length of this word.
                current_line_width = word_len
            else:
                # If the current line already contains words, calculate the required width
                # if this word is added after a space (width 1).
                required_width = current_line_width + 1 + word_len
                if required_width <= W:
                    # If the word fits, update the current line width.
                    current_line_width = required_width
                else:
                    # If the word does not fit, it must start a new line.
                    lines_needed += 1 # Increment the line count
                    # Reset the current line width to the length of this word (which starts the new line).
                    current_line_width = word_len
            
            # Optimization: If the number of lines required already exceeds M,
            # we can immediately determine that this width W is insufficient and return False.
            # This avoids unnecessary further computation.
            if lines_needed > M:
                 return False

        # If the loop completes without exceeding M lines, it means all words were placed successfully.
        # The condition lines_needed <= M is implicitly true due to the early exit optimization.
        return True

    # Determine the binary search range [low, high] for the window width W.
    
    # Lower bound `low`: The window width must be at least the length of the longest word,
    # as any line containing the longest word must have at least that width.
    low = 0
    # Iterate through word lengths to find the maximum length. Constraints state N >= 1.
    for x in L:
        low = max(low, x)

    # Upper bound `high`: A safe upper bound is the width required if all words were placed on a single line.
    # This width is the sum of all word lengths plus (N-1) spaces, each of width 1.
    # Python's integers handle arbitrary precision, so large sums are not an issue.
    high = sum(L) + (N - 1) 
    
    # Initialize `ans` to `high`. `ans` will store the minimum valid width found during the search.
    # Initializing with `high` ensures `ans` holds a valid width if the loop doesn't find better.
    ans = high 

    # Perform binary search on the possible window width W.
    while low <= high:
        # Calculate the middle width `mid` to test.
        # Using `low + (high - low) // 2` avoids potential overflow in languages with fixed-size integers,
        # and is standard practice. In Python, `(low + high) // 2` is also safe due to arbitrary precision integers.
        mid = low + (high - low) // 2
        
        # Test if the width `mid` is sufficient using the `check` function.
        if check(mid):
            # If `mid` width works, it's a potential minimum width.
            # Store it in `ans` as the best answer found so far.
            # Try to find an even smaller valid width by searching in the lower half: [low, mid-1].
            ans = mid
            high = mid - 1
        else:
            # If `mid` width does not work (too small), the minimum required width must be larger.
            # Search in the upper half: [mid+1, high].
            low = mid + 1

    # After the binary search completes, `ans` holds the minimum window width W
    # that allows the sentence to be displayed within M lines.
    print(ans)

# Execute the main function to solve the problem
solve()