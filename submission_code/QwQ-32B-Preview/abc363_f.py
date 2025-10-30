def is_palindrome(s):
    return s == s[::-1]

def get_palindromic_divisors(N):
    divisors = []
    # Check divisors from 1 to sqrt(N)
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            # Check if i is a palindromic number with digits 1-9
            s = str(i)
            if is_palindrome(s) and all(c in '123456789' for c in s):
                divisors.append(i)
            # Check if N/i is a palindromic number with digits 1-9
            s = str(N // i)
            if is_palindrome(s) and all(c in '123456789' for c in s):
                divisors.append(N // i)
    return divisors

def find_factorization(N, path=[]):
    if N == 1:
        return path
    if is_palindrome(str(N)) and all(c in '123456789' for c in str(N)):
        return path + [N]
    for A in get_palindromic_divisors(N):
        if N % (A * A) == 0:
            remainder = N // (A * A)
            sub_factorization = find_factorization(remainder, path + [A])
            if sub_factorization != -1:
                return sub_factorization + [A]
    return -1

def main():
    import sys
    N = int(sys.stdin.read())
    factorization = find_factorization(N)
    if factorization == -1:
        print(-1)
    else:
        S = '*'.join(map(str, factorization))
        # Verify that S is a palindrome and evaluates to N
        if is_palindrome(S) and eval(S) == N:
            print(S)
        else:
            print(-1)

if __name__ == "__main__":
    main()