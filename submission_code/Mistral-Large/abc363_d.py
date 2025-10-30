import sys

def find_nth_palindrome(n):
    if n == 1:
        return 0

    n -= 2  # Adjust n to find the (n-2)th palindrome since 0 and 1 are the first two palindromes

    # Determine the number of digits in the palindrome
    digits = 1
    while n >= 9 * (10 ** ((digits - 1) // 2)):
        n -= 9 * (10 ** ((digits - 1) // 2))
        digits += 2

    # Generate the first half of the palindrome
    first_half = 10 ** ((digits - 1) // 2) + n // (digits // 2)
    first_half_str = str(first_half)

    # Generate the second half of the palindrome
    if digits % 2 == 0:
        second_half_str = first_half_str[::-1]
    else:
        second_half_str = first_half_str[:-1][::-1]

    # Combine both halves to form the palindrome
    palindrome = int(first_half_str + second_half_str)
    return palindrome

def main():
    input = sys.stdin.read()
    n = int(input.strip())
    result = find_nth_palindrome(n)
    sys.stdout.write(str(result) + '
')

if __name__ == "__main__":
    main()