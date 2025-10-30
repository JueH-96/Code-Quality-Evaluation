# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    A, B = map(int, data.split())
    sum_ab = A + B
    
    # We need to print any integer between 0 and 9 that is not equal to A + B
    # Since A + B is guaranteed to be between 0 and 9, we can just iterate and find one
    for i in range(10):
        if i != sum_ab:
            print(i)
            break

if __name__ == "__main__":
    main()