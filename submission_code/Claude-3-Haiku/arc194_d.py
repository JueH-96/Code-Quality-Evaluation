from collections import deque

def count_distinct_strings(N, S):
    mod = 998244353
    
    # Initialize a queue to store the valid parenthesis sequences
    queue = deque([S])
    
    # Initialize a set to store the distinct strings
    distinct_strings = set([S])
    
    while queue:
        curr_string = queue.popleft()
        
        # Try reversing all possible substrings
        for i in range(N):
            for j in range(i, N):
                if is_valid_parenthesis(curr_string[i:j+1]):
                    new_string = curr_string[:i] + curr_string[i:j+1][::-1] + curr_string[j+1:]
                    if new_string not in distinct_strings:
                        distinct_strings.add(new_string)
                        queue.append(new_string)
    
    return len(distinct_strings) % mod

def is_valid_parenthesis(s):
    stack = []
    for c in s:
        if c == '(':
            stack.append(c)
        else:
            if not stack:
                return False
            stack.pop()
    return not stack

# Read the input
N = int(input())
S = input()

# Solve the problem and print the answer
print(count_distinct_strings(N, S))