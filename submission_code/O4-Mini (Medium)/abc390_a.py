def main():
    A = list(map(int, input().split()))
    
    # Count inversions in A
    inv_count = 0
    for i in range(5):
        for j in range(i+1, 5):
            if A[i] > A[j]:
                inv_count += 1
    
    # Exactly one adjacent swap is possible iff there is exactly one inversion
    # (and the array is not already sorted)
    if inv_count == 1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()