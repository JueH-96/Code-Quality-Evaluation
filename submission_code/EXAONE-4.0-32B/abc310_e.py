def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    c0 = 0
    c1 = 0
    total_sum = 0
    
    for char in s:
        if char == '0':
            new_c0 = 1
            new_c1 = c0 + c1
        else:
            new_c0 = c1
            new_c1 = c0 + 1
            
        total_sum += new_c1
        c0, c1 = new_c0, new_c1
        
    print(total_sum)

if __name__ == '__main__':
    main()