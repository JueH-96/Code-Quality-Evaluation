def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    # Convert string to list for modification
    boxes = list(S)
    
    # Counter for cookies eaten
    eaten = 0
    
    # Iterate from the right end to the left
    for i in range(N-1, -1, -1):
        if boxes[i] == '@':
            boxes[i] = '.'
            eaten += 1
            if eaten == D:
                break
    
    # Join the list back to a string and print
    print(''.join(boxes))

if __name__ == "__main__":
    main()