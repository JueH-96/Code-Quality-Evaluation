def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    s = input_data[1].strip()
    
    count = 0
    for i in range(n - 2):
        if s[i] == '#' and s[i+1] == '.' and s[i+2] == '#':
            count += 1

    print(count)
    
if __name__ == '__main__':
    main()