s = input().strip()
max_len = 1
n = len(s)
for i in range(n):
    for j in range(i, n):
        substr = s[i:j+1]
        if substr == substr[::-1]:
            current_length = j - i + 1
            if current_length > max_len:
                max_len = current_length
print(max_len)