# YOUR CODE HERE
def is_palindrome(num):
    return str(num) == str(num)[::-1]

def find_nth_palindrome(n):
    if n == 1:
        return 0
    
    n -= 1
    length = 1
    while True:
        start = 10**(length-1)
        end = 10**length - 1
        if n < end - start + 1:
            break
        n -= (end - start + 1)
        length += 1
    
    if length == 1:
        return n
    
    half = str(n + start)
    if length % 2 == 0:
        return int(half + half[::-1])
    else:
        return int(half + half[:-1][::-1])

n = int(input())
print(find_nth_palindrome(n))