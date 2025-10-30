# YOUR CODE HERE
def generate_palindrome(n):
    if n == 1:
        return 0
    n -= 1  # Adjust for 0-based indexing
    length = 1
    while True:
        # Generate palindromes of length `length`
        half_length = (length + 1) // 2
        start = 10**(half_length - 1)
        end = 10**half_length
        for i in range(start, end):
            if length % 2 == 0:
                palindrome = int(str(i) + str(i)[::-1])
            else:
                palindrome = int(str(i) + str(i)[:-1][::-1])
            n -= 1
            if n == 0:
                return palindrome
        length += 1

N = int(input())
print(generate_palindrome(N))