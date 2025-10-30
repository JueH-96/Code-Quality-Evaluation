def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    min_dict = {}
    
    for i in range(1, n + 1):
        line = data[i].split()
        a = int(line[0])
        c = int(line[1])
        if c in min_dict:
            if a < min_dict[c]:
                min_dict[c] = a
        else:
            min_dict[c] = a
            
    result = max(min_dict.values())
    print(result)

if __name__ == "__main__":
    main()