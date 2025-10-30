def main():
    K = int(input().strip())
    numbers = []
    for bitmask in range(1, 1 << 10):
        if bitmask == 1:
            continue
        digits = []
        for d in range(10):
            if bitmask & (1 << d):
                digits.append(d)
        digits.sort(reverse=True)
        num = 0
        for digit in digits:
            num = num * 10 + digit
        numbers.append(num)
    
    numbers.sort()
    print(numbers[K-1])

if __name__ == '__main__':
    main()