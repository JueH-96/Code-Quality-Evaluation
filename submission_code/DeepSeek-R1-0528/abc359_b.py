def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    positions = [[] for _ in range(n + 1)]
    
    for index, color in enumerate(arr):
        positions[color].append(index)
    
    count = 0
    for i in range(1, n + 1):
        if len(positions[i]) == 2:
            p1, p2 = positions[i]
            if p2 - p1 == 2:
                count += 1
                
    print(count)

if __name__ == "__main__":
    main()