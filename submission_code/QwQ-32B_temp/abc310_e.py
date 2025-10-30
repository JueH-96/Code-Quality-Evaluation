n = int(input())
s = input().strip()

prev_c0 = 0
prev_c1 = 0
total = 0

for c in s:
    a = int(c)
    if a == 1:
        trans0 = prev_c1
        trans1 = prev_c0
    else:
        trans0 = 0
        trans1 = prev_c0 + prev_c1
    
    new_c0 = trans0 + (1 if a == 0 else 0)
    new_c1 = trans1 + (1 if a == 1 else 0)
    
    total += new_c1
    prev_c0, prev_c1 = new_c0, new_c1

print(total)