def main():
    K = int(input().strip())
    numbers = []
    for mask in range(1, 1 << 10):
        if mask == 1:
            continue
        digits = []
        for i in range(10):
            if mask & (1 << i):
                digits.append(i)
        digits.sort(reverse=True)
        num = 0
        for d in digits:
            num = num * 10 + d
        numbers.append(num)
    
    numbers.sort()
    print(numbers[K-1])

if __name__ == '__main__':
    main()