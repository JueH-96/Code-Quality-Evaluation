def main():
    H = int(input().strip())
    height = 0
    day = 0
    # Each morning on day `day`, plant height is sum of 2^i for i=0..day-1.
    # We keep adding 2^day at night and increment day until height > H.
    while height <= H:
        height += 1 << day
        day += 1
    print(day)

if __name__ == "__main__":
    main()