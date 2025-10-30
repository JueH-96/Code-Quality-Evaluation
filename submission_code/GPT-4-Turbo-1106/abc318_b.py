def calculate_area(rectangles):
    covered_points = set()
    for (x1, x2, y1, y2) in rectangles:
        for x in range(x1, x2):
            for y in range(y1, y2):
                covered_points.add((x, y))
    return len(covered_points)

if __name__ == "__main__":
    N = int(input().strip())
    rectangles = [tuple(map(int, input().strip().split())) for _ in range(N)]
    area = calculate_area(rectangles)
    print(area)