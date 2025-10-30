import sys

def find_max_length_11_22_substring(s):
    n = len(s)
    max_length = 0

    for i in range(n):
        if s[i] == '/':
            left = i - 1
            right = i + 1
            length = 1

            while left >= 0 and s[left] == '1':
                length += 1
                left -= 1

            while right < n and s[right] == '2':
                length += 1
                right += 1

            if length % 2 == 1 and left == -1 and right == n:
                max_length = max(max_length, length)

    return max_length

# Read input
n = int(sys.stdin.readline().strip())
s = sys.stdin.readline().strip()

# Solve the problem
result = find_max_length_11_22_substring(s)

# Write output
print(result)