# YOUR CODE HERE
N = int(input())
dishes = [input() for _ in range(N)]

can_eat = True
for i in range(1, N):
    if dishes[i] == 'sweet' and dishes[i-1] == 'sweet':
        can_eat = False
        break

print('Yes' if can_eat else 'No')