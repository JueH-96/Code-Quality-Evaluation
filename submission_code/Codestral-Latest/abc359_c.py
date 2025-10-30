import sys
input = sys.stdin.read
data = input().split()

sx = int(data[0])
sy = int(data[1])
tx = int(data[2])
ty = int(data[3])

def calculate_toll(sx, sy, tx, ty):
    dx = abs(tx - sx)
    dy = abs(ty - sy)

    # Calculate the number of tiles crossed horizontally and vertically
    horizontal_tiles = dx // 2 + dx % 2
    vertical_tiles = dy // 2 + dy % 2

    # Total toll is the sum of tiles crossed in both directions
    total_toll = horizontal_tiles + vertical_tiles

    return total_toll

result = calculate_toll(sx, sy, tx, ty)
print(result)