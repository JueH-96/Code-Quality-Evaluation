def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Sort A in ascending order
    A_sorted = sorted(A)
    # Sort B in ascending order
    B_sorted = sorted(B)
    
    # Check if it's possible to assign boxes to people
    possible = True
    for i in range(M):
        if A_sorted[N - M + i] < B_sorted[i]:
            possible = False
            break
    
    if not possible:
        print(-1)
        return
    
    # Now, find the minimal total cost
    # We need to select M boxes such that for each i, the box has at least B_sorted[i] candies
    # To minimize the total cost, we should select the smallest possible boxes that satisfy the condition
    # So, for each B_sorted[i], we find the smallest A_sorted[j] >= B_sorted[i]
    
    # Initialize pointers
    total_cost = 0
    a_ptr = 0
    b_ptr = 0
    selected = []
    
    # Iterate through B_sorted and find the corresponding A_sorted
    while b_ptr < M:
        while a_ptr < N and A_sorted[a_ptr] < B_sorted[b_ptr]:
            a_ptr += 1
        if a_ptr >= N:
            possible = False
            break
        selected.append(A_sorted[a_ptr])
        total_cost += A_sorted[a_ptr]
        a_ptr += 1
        b_ptr += 1
    
    if not possible:
        print(-1)
    else:
        print(total_cost)

if __name__ == "__main__":
    main()