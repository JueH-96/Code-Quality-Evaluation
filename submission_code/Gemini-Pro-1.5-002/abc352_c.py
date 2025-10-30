# YOUR CODE HERE
def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        giants.append(list(map(int, input().split())))

    import itertools

    max_head_height = 0
    for perm in itertools.permutations(range(n)):
        current_height = 0
        head_height = 0
        for i in perm:
            current_height += giants[i][0]
            head_height = current_height + giants[i][1] - giants[i][0]
        max_head_height = max(max_head_height, head_height)
    
    print(max_head_height)

solve()