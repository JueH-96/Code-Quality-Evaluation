# YOUR CODE HERE

N, M = map(int, input().split())
hands = list(map(int, input().split()))

aliens = [0]*N

for i in range(N):
    while hands[i] > 0 and M > 0:
        hands[i] -= 1
        M -= 1
        aliens[i] += 1

print(sum(aliens))