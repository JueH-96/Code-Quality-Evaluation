def count_palindromes_of_length(length):
    if length == 1:
        return 10
    elif length % 2 == 0:
        return 9 * (10 ** (length // 2 - 1))
    else:
        return 9 * (10 ** (length // 2))

def generate_kth_palindrome_of_length(k, length):
    if length == 1:
        return k - 1
    
    k -= 1  # Convert to 0-indexed
    
    if length % 2 == 0:
        half_length = length // 2
        num_combinations_per_first_digit = 10 ** (half_length - 1)
        first_digit = k // num_combinations_per_first_digit + 1
        remaining = k % num_combinations_per_first_digit
        
        first_half = [first_digit]
        for i in range(half_length - 2, -1, -1):
            digit = remaining // (10 ** i)
            first_half.append(digit)
            remaining %= (10 ** i)
        
        palindrome = first_half + first_half[::-1]
        return int(''.join(map(str, palindrome)))
    else:
        half_length = length // 2
        num_combinations_per_first_digit = 10 ** half_length
        first_digit = k // num_combinations_per_first_digit + 1
        remaining = k % num_combinations_per_first_digit
        
        first_half = [first_digit]
        for i in range(half_length - 1, -1, -1):
            digit = remaining // (10 ** i)
            first_half.append(digit)
            remaining %= (10 ** i)
        
        palindrome = first_half + first_half[-2::-1]
        return int(''.join(map(str, palindrome)))

def find_nth_palindrome(n):
    count = 0
    length = 1
    while True:
        length_count = count_palindromes_of_length(length)
        if count + length_count >= n:
            position = n - count
            return generate_kth_palindrome_of_length(position, length)
        count += length_count
        length += 1

n = int(input())
print(find_nth_palindrome(n))