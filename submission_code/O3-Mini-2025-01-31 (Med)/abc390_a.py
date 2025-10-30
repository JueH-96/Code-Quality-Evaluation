def main():
    import sys
    input_line = sys.stdin.read().strip().split()
    if not input_line:
        return
    # Convert the input to list of ints
    A = list(map(int, input_line))
    
    # Check all adjacent swap positions
    target = [1, 2, 3, 4, 5]
    for i in range(len(A) - 1):
        swapped = A.copy()
        swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
        if swapped == target:
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()