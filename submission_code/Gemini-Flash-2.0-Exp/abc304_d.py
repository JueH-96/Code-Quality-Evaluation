def solve():
    W, H = map(int, input().split())
    N = int(input())
    strawberries = []
    for _ in range(N):
        strawberries.append(tuple(map(int, input().split())))
    A = int(input())
    a = list(map(int, input().split()))
    B = int(input())
    b = list(map(int, input().split()))

    counts = {}
    for i in range(A + 1):
        for j in range(B + 1):
            counts[(i, j)] = 0

    for p, q in strawberries:
        x_index = 0
        for i in range(A):
            if p > a[i]:
                x_index += 1
            else:
                break

        y_index = 0
        for i in range(B):
            if q > b[i]:
                y_index += 1
            else:
                break
        
        counts[(x_index, y_index)] += 1
    
    min_strawberries = float('inf')
    max_strawberries = float('-inf')

    for count in counts.values():
        min_strawberries = min(min_strawberries, count)
        max_strawberries = max(max_strawberries, count)
    
    print(min_strawberries, max_strawberries)

solve()