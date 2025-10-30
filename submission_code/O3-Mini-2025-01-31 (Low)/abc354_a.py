def main():
    import sys
    input = sys.stdin.readline
    
    H = int(input().strip())
    day = 0
    height = 0  # height at morning of current day
    
    # Since the plant height at morning of day i is sum_{j=0}^{i-1} 2^j = 2^i - 1,
    # we find the smallest i such that (2^i - 1) > H.
    # We can simulate by doubling and adding.
    
    while height <= H:
        # Increase day count to represent going to next morning.
        day += 1
        height = 2 ** day - 1
        
    print(day)

if __name__ == '__main__':
    main()