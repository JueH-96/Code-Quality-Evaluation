# YOUR CODE HERE
def count_special_seats(N, S):
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i+1] == '.' and S[i+2] == '#':
            count += 1
    return count

# Read input
N = int(input())
S = input().strip()

# Calculate and print the result
result = count_special_seats(N, S)
print(result)