def main():
    import sys
    input = sys.stdin.readline

    # Read input from STDIN
    N = int(input().strip())
    arr = list(map(int, input().split()))
    
    # Determine the largest integer in the list
    max_value = max(arr)
    
    # Filter out the maximum value elements and then take max amongst the rest
    # The problem guarantees that the list of remaining elements is non-empty.
    filtered = [x for x in arr if x != max_value]
    print(max(filtered))

if __name__ == "__main__":
    main()