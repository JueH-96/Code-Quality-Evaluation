N, A = map(int, input().split())
T = list(map(int, input().split()))

finish_time = 0  # Time when current person finishes
result = []

for i in range(N):
    # If current person arrives after previous person finishes
    if T[i] >= finish_time:
        finish_time = T[i] + A
    # If current person arrives while someone is being served/in queue
    else:
        finish_time += A
        
    result.append(finish_time)

for x in result:
    print(x)