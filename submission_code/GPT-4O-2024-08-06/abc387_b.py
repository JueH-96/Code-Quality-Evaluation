# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    X = int(input().strip())
    
    total_sum = 0
    count_X = 0
    
    for i in range(1, 10):
        for j in range(1, 10):
            product = i * j
            total_sum += product
            if product == X:
                count_X += 1
    
    # Subtract the sum of all X's from the total sum
    result = total_sum - (X * count_X)
    
    print(result)

if __name__ == "__main__":
    main()