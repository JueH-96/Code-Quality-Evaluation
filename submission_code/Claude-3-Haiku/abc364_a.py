# Read the input
N = int(input())
dishes = [input() for _ in range(N)]

# Check if Takahashi can eat all the dishes
consecutive_sweet = 0
for dish in dishes:
    if dish == 'sweet':
        consecutive_sweet += 1
        if consecutive_sweet >= 2:
            print('No')
            return
    else:
        consecutive_sweet = 0

print('Yes')