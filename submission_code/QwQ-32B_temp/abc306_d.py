import sys

def main():
    N = int(sys.stdin.readline())
    courses = []
    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        courses.append((X, Y))
    
    prev_healthy = 0
    prev_upset = -float('inf')
    
    for X, Y in courses:
        new_h = -float('inf')
        new_u = -float('inf')
        
        # Skip option
        new_h = max(new_h, prev_healthy)
        new_u = max(new_u, prev_upset)
        
        # Eat option
        if X == 0:
            temp1 = prev_healthy + Y
            temp2 = prev_upset + Y
            candidate = max(temp1, temp2)
            if candidate > new_h:
                new_h = candidate
        else:
            candidate = prev_healthy + Y
            if candidate > new_u:
                new_u = candidate
        
        prev_healthy, prev_upset = new_h, new_u
    
    ans = max(max(prev_healthy, prev_upset), 0)
    print(ans)

if __name__ == "__main__":
    main()