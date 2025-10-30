def is_palindrome(s):
    return s == s[::-1]

def can_form_palindrome(strings):
    n = len(strings)
    for i in range(n):
        for j in range(n):
            if i != j:
                if is_palindrome(strings[i] + strings[j]):
                    return True
    return False

import sys
input = sys.stdin.read

def main():
    data = input().strip().split('
')
    N = int(data[0])
    strings = data[1:N+1]
    
    if can_form_palindrome(strings):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()