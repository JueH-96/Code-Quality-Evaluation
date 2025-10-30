def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    giants = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        giants.append((A, B))
        index += 2
    
    # Sort giants by decreasing A
    giants.sort(reverse=True, key=lambda x: x[0])
    
    # Calculate the total height
    total_height = sum(giant[0] for giant in giants[:-1]) + giants[-1][1]
    
    print(total_height)

if __name__ == '__main__':
    main()