import math

def find_palindrome_factors(n):
    # Check if n is a palindrome itself
    if str(n) == str(n)[::-1]:
        return str(n)
    
    # Check for palindromic factors
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factor1, factor2 = i, n // i
            # Check if both factors are palindromes and do not contain '0'
            if str(factor1) == str(factor1)[::-1] and str(factor2) == str(factor2)[::-1] and '0' not in str(factor1) and '0' not in str(factor2):
                # Check if the concatenation of the factors is a palindrome
                combined = str(factor1) + '*' + str(factor2)
                if combined == combined[::-1]:
                    return combined
    return -1

def main():
    N = int(input().strip())
    result = find_palindrome_factors(N)
    print(result)

if __name__ == "__main__":
    main()