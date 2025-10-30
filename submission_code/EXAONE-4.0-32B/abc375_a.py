def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    count = 0
    for i in range(len(s) - 2):
        if s[i] == '#' and s[i+1] == '.' and s[i+2] == '#':
            count += 1
            
    print(count)

if __name__ == "__main__":
    main()