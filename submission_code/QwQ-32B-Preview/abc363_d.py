def find_nth_palindrome(N):
    if N == 1:
        return 0
    cumulative = 0
    l = 1
    while True:
        if l == 1:
            count_l = 10
        else:
            if l % 2 == 0:
                count_l = 9 * 10**((l // 2) - 1)
            else:
                count_l = 9 * 10**(l // 2)
        if cumulative + count_l >= N:
            break
        cumulative += count_l
        l += 1
    pos = N - cumulative
    h = (l + 1) // 2
    if l == 1:
        start = 0
    else:
        start = 10**(h - 1)
    prefix_number = start + pos - 1
    prefix_str = str(prefix_number)
    if l % 2 == 0:
        suffix_str = prefix_str[::-1]
    else:
        suffix_str = prefix_str[:-1][::-1]
    palindrome_str = prefix_str + suffix_str
    return int(palindrome_str)

# Read N from input
N = int(input())
# Find and print the N-th palindrome
print(find_nth_palindrome(N))