# YOUR CODE HERE
Q = int(input())
ball_count = {}

for _ in range(Q):
    query = input().split()
    
    if query[0] == '1':
        x = int(query[1])
        ball_count[x] = ball_count.get(x, 0) + 1
    
    elif query[0] == '2':
        x = int(query[1])
        ball_count[x] -= 1
        if ball_count[x] == 0:
            del ball_count[x]
    
    elif query[0] == '3':
        print(len(ball_count))