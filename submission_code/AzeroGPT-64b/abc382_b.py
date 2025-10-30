N, D = map(int, input().split())
S = input()

# Create a list to represent the result
result = ['.' for _ in range(N)]

# Count the remaining cookies after D days
remaining_cookie_position = N - S[::-1].index('@') - 1
for i in range(N):
    if remaining_cookie_position > D:
        result[N-i-1] = '@'
    else:
        if N-i-1 >= remaining_cookie_position:
            result[N-i-1] = '@'
            remaining_cookie_position = N - S[:N-i-1][::-1].index('@') - 1

# Print the result
print(''.join(result))