n = int(input())
s = input().strip()

total = 0
prev0 = 0
prev1 = 0

for c in s:
    current = int(c)
    if current == 0:
        term = prev1 * 1
    else:
        term = 0
    count_1_prev = prev0 + (prev1 * (1 if current == 0 else 0))
    count_0_prev = prev1 * (1 if current == 1 else 0)
    
    new0 = count_0_prev + (1 if current == 0 else 0)
    new1 = count_1_prev + (1 if current == 1 else 0)
    
    total += new1
    prev0, prev1 = new0, new1

print(total)