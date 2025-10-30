def main():
    N = int(input().strip())
    A = list(map(int, input().strip().split()))
    
    max_val = max(A)
    # Filter out all occurrences of the largest value
    filtered = [x for x in A if x != max_val]
    
    # Print the largest value among the remaining
    print(max(filtered))

main()