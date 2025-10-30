def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    balls = list(map(int, input_data))
    
    count = {}
    for color in balls:
        count[color] = count.get(color, 0) + 1
    
    operations = 0
    for freq in count.values():
        operations += freq // 2
    
    print(operations)

if __name__ == '__main__':
    main()