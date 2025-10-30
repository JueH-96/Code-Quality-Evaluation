import sys

def solve():
    S = sys.stdin.readline().strip()

    stack = []
    
    # Mapping of closing brackets to their corresponding opening brackets
    # This allows efficient checking of matching pairs.
    bracket_map = {
        ')': '(',
        ']': '[',
        '>': '<'
    }
    
    # Set of all opening brackets for quick lookup
    opening_brackets = {'(', '[', '<'}

    for char in S:
        if char in opening_brackets:
            # If the character is an opening bracket, push it onto the stack.
            stack.append(char)
        else:
            # If the character is a closing bracket:
            # 1. Check if the stack is empty. If it is, there's no opening bracket to match.
            if not stack:
                print("No")
                return
            
            # 2. Get the expected opening bracket for the current closing bracket.
            #    e.g., if char is ')', expected_open is '('.
            expected_open = bracket_map[char]
            
            # 3. Pop the top element from the stack. This is the last encountered opening bracket.
            last_open = stack.pop()
            
            # 4. Check if the popped opening bracket matches the expected one.
            #    If not, the brackets are mismatched (e.g., ([)]).
            if last_open != expected_open:
                print("No")
                return
        
    # After iterating through all characters:
    # If the stack is empty, it means all opening brackets found their matches and were "deleted".
    # Otherwise, there are unmatched opening brackets.
    if not stack:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the program
solve()