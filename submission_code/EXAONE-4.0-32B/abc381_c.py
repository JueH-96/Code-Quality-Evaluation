def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    if n == 0:
        print(0)
        return
        
    left_ones = [0] * n
    for i in range(1, n):
        if s[i-1] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0
            
    right_twos = [0] * n
    for i in range(n-2, -1, -1):
        if s[i+1] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
            
    best = 0
    for i in range(n):
        if s[i] == '/':
            x = min(left_ones[i], right_twos[i])
            length = 2 * x + 1
            if length > best:
                best = length
                
    print(best)

if __name__ == "__main__":
    main()