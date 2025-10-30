def main():
    n = int(input().strip())
    low = (n // 5) * 5
    high = low + 5
    
    if high > 100:
        print(low)
    else:
        dist_low = abs(n - low)
        dist_high = abs(n - high)
        if dist_low < dist_high:
            print(low)
        else:
            print(high)

if __name__ == "__main__":
    main()