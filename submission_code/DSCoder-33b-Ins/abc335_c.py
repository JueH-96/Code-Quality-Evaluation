import sys

N, Q = map(int, input().split())

# Initialize the dragon parts
dragon = [(i, 0) for i in range(N)]

# Process the queries
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        # Move the head
        if query[1] == 'R':
            dragon[0] = (dragon[0][0]+1, dragon[0][1])
        elif query[1] == 'L':
            dragon[0] = (dragon[0][0]-1, dragon[0][1])
        elif query[1] == 'U':
            dragon[0] = (dragon[0][0], dragon[0][1]+1)
        elif query[1] == 'D':
            dragon[0] = (dragon[0][0], dragon[0][1]-1)
        # Move the rest of the body
        for i in range(1, N):
            dx = dragon[i-1][0] - dragon[i][0]
            dy = dragon[i-1][1] - dragon[i][1]
            if abs(dx) > 1 or abs(dy) > 1:
                dragon[i] = (dragon[i][0] + dx//abs(dx) if dx else 0, dragon[i][1] + dy//abs(dy) if dy else 0)
    elif query[0] == '2':
        # Print the coordinates of the part
        print(*dragon[int(query[1])-1])