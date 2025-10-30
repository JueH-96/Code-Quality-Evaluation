H, W, N = map(int, input().split())
holes = set()
for _ in range(N):
    a, b = map(int, input().split())
    holes.add((a-1, b-1))  # Convert to 0-based indexing

def is_holeless_square(i, j, n):
    if i + n > H or j + n > W:
        return False
    
    for di in range(n):
        for dj in range(n):
            if (i + di, j + dj) in holes:
                return False
    return True

answer = 0
for i in range(H):
    for j in range(W):
        n = 1
        while is_holeless_square(i, j, n):
            answer += 1
            n += 1

print(answer)