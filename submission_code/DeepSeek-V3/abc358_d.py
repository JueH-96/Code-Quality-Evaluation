# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))
    
    # Sort A and B
    A_sorted = sorted(A)
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
    # We need to select M boxes such that for each i, A_j >= B_i
    # To minimize the total cost, we should select the smallest possible A_j >= B_i
    # So, we can assign the smallest A_j >= B_i to each B_i in order
    
    # Initialize pointers
    a_ptr = 0
    b_ptr = 0
    total_cost = 0
    selected = []
    
    # Iterate through B_sorted and find the smallest A_sorted >= B_sorted[i]
    for b in B_sorted:
        while a_ptr < N and A_sorted[a_ptr] < b:
            a_ptr += 1
        if a_ptr >= N:
            possible = False
            break
        selected.append(A_sorted[a_ptr])
        a_ptr += 1
    
    if not possible:
        print(-1)
    else:
        print(sum(selected))

if __name__ == "__main__":
    main()