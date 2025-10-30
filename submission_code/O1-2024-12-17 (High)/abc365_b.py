# YOUR CODE HERE
def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Find the second largest value
    second_largest = sorted(A, reverse=True)[1]
    
    # Get the 1-based index of the second largest value in the original list
    idx = A.index(second_largest) + 1
    
    print(idx)

# Do not forget to call main() at the end!
if __name__ == "__main__":
    main()