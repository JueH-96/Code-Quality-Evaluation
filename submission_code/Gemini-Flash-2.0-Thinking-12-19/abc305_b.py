def solve():
    p, q = input().split()
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    distances = [3, 1, 4, 1, 5, 9]
    index_p = -1
    index_q = -1
    for i in range(len(points)):
        if points[i] == p:
            index_p = i
        if points[i] == q:
            index_q = i
    
    if index_p > index_q:
        index_p, index_q = index_q, index_p
        
    total_distance = 0
    for i in range(index_p, index_q):
        total_distance += distances[i]
        
    print(total_distance)

if __name__ == '__main__':
    solve()