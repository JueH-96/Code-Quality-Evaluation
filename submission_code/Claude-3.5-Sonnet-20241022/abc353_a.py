N = int(input())
H = list(map(int, input().split()))

first_height = H[0]
answer = -1

for i in range(1, N):
    if H[i] > first_height:
        answer = i + 1
        break

print(answer)