S = input().strip()

def is_palindrome(s):
    return s == s[::-1]

max_length = 1

for i in range(len(S)):
    for j in range(i, len(S)):
        substring = S[i:j+1]
        if is_palindrome(substring):
            max_length = max(max_length, len(substring))

print(max_length)