n = int(input())
binary = bin(n)[2:][::-1]
count = 0
for c in binary:
    if c == '0':
        count += 1
    else:
        break
print(count)