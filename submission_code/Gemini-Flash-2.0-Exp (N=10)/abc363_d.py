def is_palindrome(n):
    s = str(n)
    return s == s[::-1]

def generate_palindromes():
    palindromes = [0]
    for i in range(1, 10):
        palindromes.append(i)
    
    for length in range(2, 20):
        for first_digit in range(1, 10):
            if length % 2 == 0:
                mid_len = length // 2
                for mid in range(10 ** (mid_len - 1) if mid_len > 1 else 0, 10 ** mid_len):
                    s = str(first_digit) + str(mid) + str(mid)[::-1]
                    palindromes.append(int(s))
            else:
                mid_len = (length - 1) // 2
                for mid in range(10 ** mid_len if mid_len > 0 else 0, 10 ** (mid_len + 1)):
                    s = str(first_digit) + str(mid)[:-1] + str(mid)[::-1]
                    palindromes.append(int(s))
    palindromes.sort()
    return palindromes

n = int(input())
palindromes = generate_palindromes()
print(palindromes[n])