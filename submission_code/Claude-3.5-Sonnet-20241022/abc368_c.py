N = int(input())
H = list(map(int, input().split()))

T = 0
curr_enemy = 0

while curr_enemy < N:
    T += 1
    if H[curr_enemy] > 0:
        if T % 3 == 0:
            H[curr_enemy] -= 3
        else:
            H[curr_enemy] -= 1
            
    while curr_enemy < N and H[curr_enemy] <= 0:
        curr_enemy += 1

print(T)