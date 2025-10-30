# YOUR CODE HERE
def count_empty_boxes(N, D, S):
    cookie_count = S.count('@')
    return N - (cookie_count - D)

# Read input
N, D = map(int, input().split())
S = input().strip()

# Calculate and print the result
result = count_empty_boxes(N, D, S)
print(result)