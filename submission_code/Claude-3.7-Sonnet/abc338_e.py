def main():
    N = int(input().strip())
    chords = []
    
    for _ in range(N):
        a, b = map(int, input().strip().split())
        chords.append((a, b))
    
    # Check for intersections
    has_intersection = False
    for i in range(N):
        for j in range(i+1, N):
            a, b = chords[i]
            c, d = chords[j]
            if do_chords_intersect(a, b, c, d, N):
                has_intersection = True
                break
        if has_intersection:
            break
    
    print("Yes" if has_intersection else "No")

def do_chords_intersect(a, b, c, d, N):
    # Function to check if point p is inside arc (x, y)
    def is_inside_arc(x, y, p):
        # Clockwise distance from x to y
        dist_x_to_y = (y - x) % (2*N)
        # Clockwise distance from x to p
        dist_x_to_p = (p - x) % (2*N)
        # p is inside arc (x, y) if 0 < dist_x_to_p < dist_x_to_y
        return 0 < dist_x_to_p < dist_x_to_y
    
    # Check if exactly one of the points c or d is inside the arc (a, b)
    c_inside = is_inside_arc(a, b, c)
    d_inside = is_inside_arc(a, b, d)
    
    return c_inside != d_inside

if __name__ == "__main__":
    main()