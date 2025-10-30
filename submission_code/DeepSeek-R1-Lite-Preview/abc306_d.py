import sys

def main():
    import math
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    courses = []
    idx = 1
    for _ in range(N):
        X = int(data[idx])
        Y = int(data[idx + 1])
        courses.append((X, Y))
        idx += 2
    
    prev_healthy = 0
    prev_upset = -math.inf
    
    for X, Y in courses:
        if X == 0:
            eat_to_healthy = max(prev_healthy + Y, prev_upset + Y)
            current_healthy = max(prev_healthy, eat_to_healthy)
            current_upset = prev_upset
        else:
            eat_to_upset = prev_healthy + Y
            current_healthy = prev_healthy
            current_upset = max(prev_upset, eat_to_upset)
        
        prev_healthy, prev_upset = current_healthy, current_upset
    
    answer = max(prev_healthy, prev_upset, 0)
    print(answer)

if __name__ == "__main__":
    main()