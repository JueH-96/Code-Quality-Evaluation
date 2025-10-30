import sys

def is_palindrome(s):
    return s == s[::-1]

def can_form_palindrome(strings):
    n = len(strings)
    for i in range(n):
        for j in range(n):
            if i != j:
                concatenated = strings[i] + strings[j]
                if is_palindrome(concatenated):
                    return True
    return False

def main():
    input = sys.stdin.read
    data = input().split()

    n = int(data[0])
    strings = data[1:n+1]

    if can_form_palindrome(strings):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()