N = int(input())

M = (N - 1).bit_length()

print(M)

for bit in range(M):
    bottles = []
    for bottle in range(1, N + 1):
        if (bottle - 1) & (1 << bit):
            bottles.append(bottle)
    
    if bottles:
        print(len(bottles), *bottles)
    else:
        print(0)

S = input().strip()

spoiled = 0
for i in range(M):
    if S[i] == '1':
        spoiled |= (1 << i)

spoiled += 1
print(spoiled)