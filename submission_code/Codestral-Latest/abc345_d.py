from itertools import permutations

def can_cover_grid(N, H, W, tiles):
    total_area = H * W
    for perm in permutations(tiles):
        used_area = 0
        for a, b in perm:
            if used_area + a * b > total_area:
                break
            used_area += a * b
            if used_area == total_area:
                return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    H = int(data[index + 1])
    W = int(data[index + 2])
    index += 3

    tiles = []
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        tiles.append((A, B))
        index += 2

    if can_cover_grid(N, H, W, tiles):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()