def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    hands = list(map(int, input_data[2:2+N]))
    
    count = 0
    for h in hands:
        if M >= h:
            count += 1
            M -= h
        else:
            break
    print(count)

if __name__ == '__main__':
    main()