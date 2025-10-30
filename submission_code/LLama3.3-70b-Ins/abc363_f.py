import sys

def find_palindrome(n):
    # Check if n is a palindrome
    if str(n) == str(n)[::-1] and '0' not in str(n):
        return str(n)

    # Try to find a palindrome with a multiplication symbol
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            j = n // i
            # Check if the concatenation of i and j is a palindrome
            concat = str(i) + '*' + str(j)
            if concat == concat[::-1] and '0' not in concat:
                return concat
            # Check if the concatenation of j and i is a palindrome
            concat = str(j) + '*' + str(i)
            if concat == concat[::-1] and '0' not in concat:
                return concat

    # Try to find a palindrome with two multiplication symbols
    for i in range(1, int(n ** (1/3)) + 1):
        if n % i == 0:
            j = n // i
            for k in range(i, int(j ** 0.5) + 1):
                if j % k == 0:
                    l = j // k
                    # Check if the concatenation of i, k, and l is a palindrome
                    concat = str(i) + '*' + str(k) + '*' + str(l)
                    if concat == concat[::-1] and '0' not in concat:
                        return concat
                    # Check if the concatenation of l, k, and i is a palindrome
                    concat = str(l) + '*' + str(k) + '*' + str(i)
                    if concat == concat[::-1] and '0' not in concat:
                        return concat

    # If no palindrome is found, return -1
    return -1

n = int(input())
print(find_palindrome(n))