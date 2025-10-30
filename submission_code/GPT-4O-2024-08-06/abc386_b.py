# YOUR CODE HERE
def min_button_presses(S):
    n = len(S)
    presses = 0
    i = 0
    
    while i < n:
        if i < n - 1 and S[i] == '0' and S[i + 1] == '0':
            # Use the '00' button
            presses += 1
            i += 2  # Skip the next character as well
        else:
            # Use the single character button
            presses += 1
            i += 1
    
    return presses

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(min_button_presses(S))