n = int(input())
s = input()

max_length = 1  # At least one '/' exists, so minimum is 1

for i in range(n):
    if s[i] == '/':
        # Try to expand as much as possible
        expansion = 0
        while True:
            left = i - expansion - 1
            right = i + expansion + 1
            
            if left >= 0 and right < n and s[left] == '1' and s[right] == '2':
                expansion += 1
            else:
                break
        
        # The length is 1 + 2 * expansion
        current_length = 1 + 2 * expansion
        max_length = max(max_length, current_length)

print(max_length)