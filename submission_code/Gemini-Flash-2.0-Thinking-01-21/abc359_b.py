def solve():
    n = int(input())
    a = list(map(int, input().split()))
    color_indices = {}
    for i in range(2 * n):
        color = a[i]
        if color not in color_indices:
            color_indices[color] = []
        color_indices[color].append(i)
    
    count = 0
    for color in range(1, n + 1):
        indices = color_indices[color]
        pos1 = indices[0]
        pos2 = indices[1]
        if abs(pos1 - pos2) == 2:
            count += 1
            
    print(count)

if __name__ == '__main__':
    solve()