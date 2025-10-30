import sys

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_nth_palindrome(n):
    i = 0
    while True:
        if is_palindrome(i):
            n -= 1
            if n == 0:
                return i
        i += 1

def main():
    n = int(sys.stdin.readline().strip())
    print(find_nth_palindrome(n))

if __name__ == "__main__":
    main()