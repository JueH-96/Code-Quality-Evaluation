import sys

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def count_integer_pairs(N, D, points):
    count = 0
    for x in range(-D, D + 1):
        for y in range(-D, D + 1):
            total_distance = sum(manhattan_distance(x, y, px, py) for px, py in points)
            if total_distance <= D:
                count += 1
    return count

def main():
    input_data = sys.stdin.read().splitlines()
    N, D = map(int, input_data[0].split())
    points = [tuple(map(int, line.split())) for line in input_data[1:]]
    result = count_integer_pairs(N, D, points)
    print(result)

if __name__ == "__main__":
    main()