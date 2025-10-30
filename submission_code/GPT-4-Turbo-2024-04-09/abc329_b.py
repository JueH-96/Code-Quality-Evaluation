def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    max_value = max(A)
    second_largest = -1
    
    for value in A:
        if value != max_value:
            if value > second_largest:
                second_largest = value
    
    print(second_largest)

if __name__ == "__main__":
    main()