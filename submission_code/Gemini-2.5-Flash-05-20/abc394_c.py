import sys

def solve():
    S = sys.stdin.readline().strip()
    
    # Use a list as a stack. Python lists are efficient for append() and pop() at the end.
    result_chars = []
    
    for char in S:
        result_chars.append(char)
        
        # This loop performs replacements as patterns form at the end of `result_chars`.
        # If a 'WA' is formed by the newly appended character and the preceding one,
        # it is replaced by 'AC'. The 'A' from the new 'AC' might then combine
        # with a character further left on the stack, forming another 'WA',
        # which this loop correctly handles by repeating the check.
        while len(result_chars) >= 2 and \
              result_chars[-2] == 'W' and \
              result_chars[-1] == 'A':
            
            result_chars.pop() # remove 'A'
            result_chars.pop() # remove 'W'
            
            result_chars.append('A')
            result_chars.append('C')
            
    sys.stdout.write("".join(result_chars) + "
")

# Call the solve function to read input and produce output
solve()