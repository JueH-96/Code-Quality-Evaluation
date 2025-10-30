import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    points = [(int(data[i*2+1]), int(data[i*2+2])) for i in range(N)]
    
    total_distance = 0
    
    for i in range(N-1):
        x1, y1 = points[i]
        for j in range(i+1, N):
            x2, y2 = points[j]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            if (x1 + y1) % 2 != (x2 + y2) % 2:
                # They are on different parities, cannot reach each other
                distance = 0
            else:
                # The distance is the maximum of the differences in x and y
                distance = max(dx, dy)
            
            total_distance += distance
    
    print(total_distance)

main()