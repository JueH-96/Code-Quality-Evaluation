# YOUR CODE HERE

def max_cube_root(n):
    low = 1
    high = 1000000  # Maximum possible x since 1000000^3 = 10^18
    while low <= high:
        mid = (low + high) // 2
        if mid ** 3 <= n:
            low = mid + 1
        else:
            high = mid - 1
    return high  # high is the largest integer x with x^3 <= n

# Read input from stdin
N = int(input())

# Find the maximum x such that x^3 <= N
max_x = max_cube_root(N)

# Iterate from max_x down to 1 to find the largest palindromic cube
for x in range(max_x, 0, -1):
    cube = x ** 3
    s = str(cube)
    if s == s[::-1]:  # Check if the string is a palindrome
        print(cube)  # Output the palindromic cube
        break  # Exit the loop after finding and printing the answer