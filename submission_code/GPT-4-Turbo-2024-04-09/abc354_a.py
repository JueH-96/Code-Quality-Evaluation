def main():
    import sys
    input = sys.stdin.read
    H = int(input().strip())
    
    height = 0
    day = 0
    while height <= H:
        height += 2 ** day
        day += 1
    
    print(day)

if __name__ == "__main__":
    main()