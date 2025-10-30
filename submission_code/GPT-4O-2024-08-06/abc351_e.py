# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = []
    
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    
    total_distance = 0
    
    for i in range(N):
        x1, y1 = points[i]
        for j in range(i + 1, N):
            x2, y2 = points[j]
            if (x1 + y1) % 2 == (x2 + y2) % 2:
                total_distance += max(abs(x2 - x1), abs(y2 - y1))
    
    print(total_distance)

if __name__ == "__main__":
    main()