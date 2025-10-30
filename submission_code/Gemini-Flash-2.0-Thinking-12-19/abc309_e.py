import collections

def solve():
    n, m = map(int, input().split())
    if n == 1:
        parents = []
    else:
        parents = list(map(int, input().split()))
    insurances = []
    for _ in range(m):
        insurances.append(tuple(map(int, input().split())))
    
    children_list = collections.defaultdict(list)
    for i in range(len(parents)):
        parent = parents[i]
        child = i + 2
        children_list[parent].append(child)
        
    is_covered = [False] * (n + 1)
    
    for insurance in insurances:
        x_i, y_i = insurance
        queue = collections.deque([(x_i, 0)])
        while queue:
            person, generation = queue.popleft()
            if generation > y_i:
                continue
            is_covered[person] = True
            if generation < y_i:
                for child in children_list[person]:
                    queue.append((child, generation + 1))
                    
    count = 0
    for i in range(1, n + 1):
        if is_covered[i]:
            count += 1
            
    print(count)

if __name__ == '__main__':
    solve()