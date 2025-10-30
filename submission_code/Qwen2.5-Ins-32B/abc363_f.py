import sys

def find_palindrome_expression(N):
    if N == 1:
        return "1"
    if N == 2:
        return "2"
    if N == 3:
        return "3"
    if N == 4:
        return "4"
    if N == 5:
        return "5"
    if N == 6:
        return "6"
    if N == 7:
        return "7"
    if N == 8:
        return "8"
    if N == 9:
        return "9"
    
    for i in range(2, int(N**0.5) + 1):
        if N % i == 0:
            j = N // i
            if i > 1 and j > 1 and str(i)[0] != '0' and str(j)[0] != '0':
                return f"{i}*{j}"
    return "-1"

N = int(input())
print(find_palindrome_expression(N))