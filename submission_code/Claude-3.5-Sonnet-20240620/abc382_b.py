# YOUR CODE HERE
def solve_cookie_problem(N, D, S):
    cookies = [i for i, char in enumerate(S) if char == '@']
    cookies.reverse()  # Start from the rightmost cookie
    
    for _ in range(D):
        if cookies:
            cookies.pop()  # Remove the rightmost cookie
    
    result = ['.' for _ in range(N)]
    for i in cookies:
        result[i] = '@'
    
    return ''.join(result)

# Read input
N, D = map(int, input().split())
S = input().strip()

# Solve and print output
print(solve_cookie_problem(N, D, S))