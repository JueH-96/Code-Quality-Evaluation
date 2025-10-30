# YOUR CODE HERE
N = int(input())
dishes = [input().strip() for _ in range(N)]

can_eat_all = True
consecutive_sweet = 0

for dish in dishes:
    if dish == 'sweet':
        consecutive_sweet += 1
        if consecutive_sweet == 2:
            can_eat_all = False
            break
    else:
        consecutive_sweet = 0

print('Yes' if can_eat_all else 'No')