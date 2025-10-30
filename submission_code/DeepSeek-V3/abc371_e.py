# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Precompute the last occurrence of each element
    last_occurrence = {}
    for idx, num in enumerate(A):
        last_occurrence[num] = idx
    
    # Initialize the result
    result = 0
    # Initialize the current distinct count
    distinct = 0
    # Initialize the last occurrence index for each element
    last_occurrence_dict = {}
    
    for i in range(N):
        if A[i] not in last_occurrence_dict:
            distinct += 1
        else:
            # If the current element is not the last occurrence, it's still contributing to distinct count
            pass
        last_occurrence_dict[A[i]] = i
        # The number of distinct elements in the current window is 'distinct'
        # The number of subarrays ending at i is (i+1)
        # So, the total contribution is distinct * (i+1)
        result += distinct
    
    print(result)

if __name__ == "__main__":
    main()