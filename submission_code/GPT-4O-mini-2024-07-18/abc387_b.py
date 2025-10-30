def main():
    import sys
    input = sys.stdin.read
    X = int(input().strip())
    
    total_sum = 0
    count_X = 0
    
    # Calculate the sum of the multiplication table and count occurrences of X
    for i in range(1, 10):
        for j in range(1, 10):
            value = i * j
            total_sum += value
            if value == X:
                count_X += 1
    
    # Each occurrence of X should be subtracted from the total sum
    sum_not_X = total_sum - (count_X * X)
    
    print(sum_not_X)

if __name__ == "__main__":
    main()