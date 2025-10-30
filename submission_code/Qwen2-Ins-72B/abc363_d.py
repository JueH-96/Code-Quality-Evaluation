def is_palindrome(n):
    return str(n) == str(n)[::-1]

def nth_palindrome(n):
    count = 0
    i = 0
    while True:
        if is_palindrome(i):
            count += 1
            if count == n:
                return i
        i += 1
        if i > 10**18:
            i = 10**18
            while not is_palindrome(i):
                i += 1

if __name__ == '__main__':
    n = int(input())
    print(nth_palindrome(n))