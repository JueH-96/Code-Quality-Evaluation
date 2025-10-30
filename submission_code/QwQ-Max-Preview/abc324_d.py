n = int(input())
s = input().strip()

# Count the frequency of each digit in the input string
count_s = [0] * 10
for c in s:
    count_s[int(c)] += 1

max_num = 10**n - 1
max_sqrt = int(max_num**0.5)
result = 0

for i in range(max_sqrt + 1):
    square = i * i
    s_square = str(square)
    k = len(s_square)
    z = n - k
    
    if z < 0:
        continue
    
    current_count = [0] * 10
    for ch in s_square:
        current_count[int(ch)] += 1
    current_count[0] += z  # add leading zeros
    
    if current_count == count_s:
        result += 1

print(result)