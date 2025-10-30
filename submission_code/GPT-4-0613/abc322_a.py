N = int(input().strip())
S = input().strip()

index = S.find('ABC')

if index == -1:
    print(-1)
else:
    print(index+1)