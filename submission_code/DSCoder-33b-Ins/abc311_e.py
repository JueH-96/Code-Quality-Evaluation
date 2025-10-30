# YOUR CODE HERE
H, W, N = map(int, input().split())
holes = set(tuple(map(int, input().split())) for _ in range(N))

def is_hole(i, j):
    return (i, j) in holes

def count_holeless_squares(H, W, holes):
    count = 0
    for i in range(1, H+1):
        for j in range(1, W+1):
            for n in range(1, min(H-i+2, W-j+2)):
                if all(not is_hole(i+k, j+l) for k in range(n) for l in range(n)):
                    count += 1
    return count

print(count_holeless_squares(H, W, holes))