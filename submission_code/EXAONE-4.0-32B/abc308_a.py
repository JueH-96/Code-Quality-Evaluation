def main():
    data = input().split()
    numbers = list(map(int, data))
    
    for num in numbers:
        if num < 100 or num > 675:
            print("No")
            return
        if num % 25 != 0:
            print("No")
            return
            
    for i in range(len(numbers) - 1):
        if numbers[i] > numbers[i+1]:
            print("No")
            return
            
    print("Yes")

if __name__ == '__main__':
    main()