# YOUR CODE HERE

N = int(input().split()[0])
H = list(map(int, input().split()))

tallest_building = -1
for i in range(N):
    if H[i] > H[0]:
        tallest_building = i
        break

print(tallest_building + 1 if tallest_building != -1 else -1)