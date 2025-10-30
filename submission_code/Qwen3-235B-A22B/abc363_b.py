def main():
    import sys
    input = sys.stdin.read().split()
    
    n = int(input[0])
    t = int(input[1])
    p = int(input[2])
    
    l = list(map(int, input[3:3+n]))
    
    current_count = 0
    for length in l:
        if length >= t:
            current_count += 1
    
    if current_count >= p:
        print(0)
    else:
        required_days = [t - length for length in l if length < t]
        required_days.sort()
        needed = p - current_count
        print(required_days[needed - 1])

if __name__ == "__main__":
    main()