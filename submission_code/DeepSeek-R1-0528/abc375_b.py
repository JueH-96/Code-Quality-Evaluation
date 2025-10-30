import math

def main():
    n = int(input().strip())
    if n == 0:
        print(0.0)
        return
    
    x0, y0 = map(int, input().split())
    total = math.hypot(x0, y0)
    
    prev_x, prev_y = x0, y0
    for _ in range(1, n):
        x, y = map(int, input().split())
        dx = prev_x - x
        dy = prev_y - y
        total += math.hypot(dx, dy)
        prev_x, prev_y = x, y
        
    total += math.hypot(prev_x, prev_y)
    print("{:.20f}".format(total))

if __name__ == "__main__":
    main()