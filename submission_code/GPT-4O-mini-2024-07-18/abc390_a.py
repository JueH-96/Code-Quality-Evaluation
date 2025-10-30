def can_sort_with_one_swap(A):
    # Count the number of inversions
    inversions = 0
    for i in range(len(A) - 1):
        for j in range(i + 1, len(A)):
            if A[i] > A[j]:
                inversions += 1

    # We can sort the array with one swap if there is exactly one inversion
    return inversions == 1

def main():
    import sys
    input = sys.stdin.read
    A = list(map(int, input().strip().split()))
    
    # Check if we can sort A with exactly one adjacent swap
    if can_sort_with_one_swap(A):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()