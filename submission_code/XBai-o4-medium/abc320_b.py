S = input().strip()
n = len(S)
max_len = 1  # At least one character is a palindrome

for i in range(n):
    for j in range(i, n):
        substr = S[i:j+1]
        if substr == substr[::-1]:
            current_len = j - i + 1
            if current_len > max_len:
                max_len = current_len

print(max_len)