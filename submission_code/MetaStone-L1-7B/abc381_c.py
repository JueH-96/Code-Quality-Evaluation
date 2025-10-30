n = int(input())
s = input().strip()

if n == 0:
    print(0)
    exit()

max_len = 0

# Precompute left_ones: number of consecutive '1's ending at each position
left_ones = [0] * n
if s[0] == '1':
    left_ones[0] = 1
for i in range(1, n):
    if s[i] == '1':
        left_ones[i] = left_ones[i-1] + 1
    else:
        left_ones[i] = 0

# Precompute right_twos: number of consecutive '2's starting at each position
right_twos = [0] * n
if s[-1] == '2':
    right_twos[-1] = 1
for i in range(n-2, -1, -1):
    if s[i] == '2':
        right_twos[i] = right_twos[i+1] + 1
    else:
        right_twos[i] = 0

for i in range(n):
    if s[i] == '/':
        # Calculate k1: number of '1's before i
        if i > 0 and s[i-1] == '1':
            k1 = left_ones[i-1]
        else:
            k1 = 0
        
        # Calculate k2: number of '2's after i
        if i < n-1 and s[i+1] == '2':
            k2 = right_twos[i+1]
        else:
            k2 = 0
        
        k = min(k1, k2)
        start = i - k
        end = i + k
        
        # Ensure the substring is within the bounds
        if start >= 0 and end < n:
            current_len = end - start + 1
        else:
            current_len = 0
        
        if current_len > max_len:
            max_len = current_len

print(max_len)