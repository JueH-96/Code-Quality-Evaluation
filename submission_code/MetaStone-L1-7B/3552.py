def largestPalindrome(n: int, k: int) -> str:
    m = (n + 1) // 2
    max_first = '9' * m

    def generate_palindrome(first_half):
        if n % 2 == 0:
            return first_half + first_half[::-1]
        else:
            return first_half + first_half[:-1][::-1]

    for current in range(int(max_first), 0, -1):
        current_str = str(current)
        if len(current_str) != m:
            continue
        palindrome = generate_palindrome(current_str)
        if len(palindrome) != n:
            continue
        if int(palindrome) % k == 0:
            return palindrome
    return ''