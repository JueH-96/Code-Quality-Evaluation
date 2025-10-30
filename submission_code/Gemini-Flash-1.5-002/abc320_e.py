# YOUR CODE HERE
n, m = map(int, input().split())
noodles = [0] * n
queue = list(range(n))
times = [0] * n
for i in range(m):
    t, w, s = map(int, input().split())
    if len(queue) > 0:
        person = queue.pop(0)
        noodles[person] += w
        times[person] = t + s
    
    
    
    added = []
    for j in range(len(queue)):
        if times[queue[j]] <= t:
            added.append(queue.pop(j))
            
    queue = added + queue
    
    
for noodle in noodles:
    print(noodle)