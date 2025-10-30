# YOUR CODE HERE
import sys
import math

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    D = int(input[1])
    positions = [(int(input[2 + 2 * i]), int(input[3 + 2 * i])) for i in range(N)]
    
    infected = [False] * N
    infected[0] = True
    
    queue = [0]
    while queue:
        current = queue.pop(0)
        cx, cy = positions[current]
        for i in range(N):
            if not infected[i]:
                nx, ny = positions[i]
                distance = math.sqrt((cx - nx) ** 2 + (cy - ny) ** 2)
                if distance <= D:
                    infected[i] = True
                    queue.append(i)
    
    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()