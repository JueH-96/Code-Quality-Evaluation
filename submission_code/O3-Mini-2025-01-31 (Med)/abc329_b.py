def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    numbers = list(map(int, data[1:n+1]))
    
    max_value = max(numbers)
    filtered_numbers = [num for num in numbers if num != max_value]
    print(max(filtered_numbers))

if __name__ == "__main__":
    main()