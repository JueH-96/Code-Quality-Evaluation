n = int(input())
s = input()

cnt = 0
prev = ''
cnts = 0

for i, c in enumerate(s):
    if i == 0:
        prev = c
        cnts += 1
        continue
        
    if c == prev:
        cnts += 1
        continue
    cnt += (cnts+1)*(cnts)//2  # find how many repeated strings: 1+2+3+...+n=n(n+1)/2
    prev = c
    cnts = 1

if cnts != 0:
    cnt += (cnts+1)*(cnts)//2
print(cnt)