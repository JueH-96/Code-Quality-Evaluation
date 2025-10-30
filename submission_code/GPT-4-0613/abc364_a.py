N = int(input().strip())
dishes = [input().strip() for _ in range(N)]

for i in range(1, N):
    if dishes[i] == 'sweet' and dishes[i-1] == 'sweet':
        print('No')
        break
else:
    print('Yes')