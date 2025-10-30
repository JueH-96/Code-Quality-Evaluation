import sys

def check(W, N, M, L, max_L_val):
    """
    Checks if the sentence can be displayed within M lines given window width W.
    
    Args:
        W (int): The candidate window width.
        N (int): The number of words.
        M (int): The maximum allowed number of lines.
        L (list): A list of word widths.
        max_L_val (int): The maximum width of a single word in L.
    
    Returns:
        bool: True if the sentence fits within M lines, False otherwise.
    """
    # If the window width W is less than the maximum single word width,
    # it's impossible to fit even one word on a line.
    if W < max_L_val:
        return False

    lines_needed = 1
    current_line_width = 0
    words_on_current_line = 0

    for i in range(N):
        word_width = L[i]

        # Case 1: This is the first word we are trying to place on the current line.
        # This occurs either at the very beginning (first word of the sentence),
        # or after starting a new line because the previous word didn't fit.
        if words_on_current_line == 0:
            current_line_width = word_width
            words_on_current_line = 1
        # Case 2: We are trying to add a word to the current line, after a space.
        else:
            # Calculate width if this word is added (current words + space + new word)
            potential_new_line_width = current_line_width + 1 + word_width
            
            if potential_new_line_width <= W:
                # Word fits on the current line
                current_line_width = potential_new_line_width
                words_on_current_line += 1
            else:
                # Word does not fit, start a new line
                lines_needed += 1
                if lines_needed > M:
                    return False # Exceeded allowed lines

                # Start the new line with the current word
                current_line_width = word_width
                words_on_current_line = 1

    return True # All words fit within M lines

def solve():
    N, M = map(int, sys.stdin.readline().split())
    L = list(map(int, sys.stdin.readline().split()))

    # Calculate the minimum possible width (at least the widest single word)
    # This value is pre-calculated to avoid redundant calls to max() inside check().
    max_L_val = 0
    for l_val in L:
        max_L_val = max(max_L_val, l_val)

    # Binary search range for W
    # Lower bound: The widest single word must fit.
    low = max_L_val
    
    # Upper bound: All words on a single line.
    # Sum of lengths + (N-1) spaces. If N=1, (N-1) is 0 spaces.
    high = sum(L) + max(0, N - 1) 

    ans = high # Initialize answer to a value that definitely works

    # Binary search
    while low <= high:
        mid = (low + high) // 2
        if check(mid, N, M, L, max_L_val):
            ans = mid     # mid is a possible answer, try to find a smaller one
            high = mid - 1
        else:
            low = mid + 1 # mid is too small, need a larger width

    print(ans)

# Call the solve function to run the program
solve()