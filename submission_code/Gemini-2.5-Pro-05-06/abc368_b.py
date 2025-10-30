def solve():
    N = int(input())
    A = list(map(int, input().split()))

    operations = 0

    while True:
        # Count positive elements in A
        num_positive = 0
        for x in A:
            if x > 0:
                num_positive += 1
        
        # Termination condition: if A contains one or fewer positive elements
        if num_positive <= 1:
            break

        # Perform the operation:
        # 1. Sort A in descending order
        A.sort(reverse=True)
        
        # 2. Decrease both A_1 (largest, A[0]) and A_2 (second largest, A[1]) by 1.
        # Since num_positive > 1 and N >= 2 (problem constraint),
        # A[0] and A[1] are valid indices and refer to positive numbers.
        # So, A[0] and A[1] will be >= 1 before decrementing.
        A[0] -= 1
        A[1] -= 1
        
        operations += 1
        
    print(operations)

if __name__ == '__main__':
    solve()