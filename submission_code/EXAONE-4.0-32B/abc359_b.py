def main():
    n = int(input().strip())
    arr = list(map(int, input().split()))
    
    first_occurrence = [-1] * (n + 1)
    count_valid = 0
    
    for index, color in enumerate(arr):
        if first_occurrence[color] == -1:
            first_occurrence[color] = index
        else:
            if index - first_occurrence[color] == 2:
                count_valid += 1
                
    print(count_valid)

if __name__ == "__main__":
    main()