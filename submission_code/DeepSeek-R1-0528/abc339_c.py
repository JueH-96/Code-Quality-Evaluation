def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    prefix = [0]
    total = 0
    for num in A:
        total += num
        prefix.append(total)
    
    min_prefix = min(prefix)
    S_min = max(0, -min_prefix)
    current_passengers = S_min + total
    print(current_passengers)

if __name__ == "__main__":
    main()