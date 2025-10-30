# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    H = int(data[2])
    K = int(data[3])
    S = data[4]
    
    health_items = set()
    index = 5
    for _ in range(M):
        x = int(data[index])
        y = int(data[index + 1])
        health_items.add((x, y))
        index += 2
    
    x, y = 0, 0
    health = H
    
    for move in S:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        health -= 1
        
        if health < 0:
            print("No")
            return
        
        if (x, y) in health_items and health < K:
            health = K
            health_items.remove((x, y))
    
    print("Yes")

if __name__ == "__main__":
    main()