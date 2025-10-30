def solve():
    W, H = map(int, input().split())
    N = int(input())
    strawberries = []
    for _ in range(N):
        strawberries.append(tuple(map(int, input().split())))
    A = int(input())
    a_cuts = list(map(int, input().split()))
    B = int(input())
    b_cuts = list(map(int, input().split()))

    a_cuts = [0] + a_cuts + [W]
    b_cuts = [0] + b_cuts + [H]

    counts = {}
    for p, q in strawberries:
        x_idx = 0
        for i in range(1, len(a_cuts)):
            if a_cuts[i-1] < p < a_cuts[i]:
                x_idx = i - 1
                break
        y_idx = 0
        for i in range(1, len(b_cuts)):
            if b_cuts[i-1] < q < b_cuts[i]:
                y_idx = i - 1
                break
        
        if (x_idx, y_idx) not in counts:
            counts[(x_idx, y_idx)] = 0
        counts[(x_idx, y_idx)] += 1

    min_strawberries = float('inf')
    max_strawberries = float('-inf')

    if len(counts) < (A + 1) * (B + 1):
        min_strawberries = 0
    else:
        min_strawberries = min(counts.values())
    
    if len(counts) > 0:
        max_strawberries = max(counts.values())
    else:
        max_strawberries = 0

    print(min_strawberries, max_strawberries)

solve()