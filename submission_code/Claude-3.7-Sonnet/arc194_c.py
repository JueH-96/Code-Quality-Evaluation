def min_cost_to_match(A, B, C):
    N = len(A)
    A = A.copy()  # Make a copy to avoid modifying the original list
    
    P01 = []  # Positions where A_i = 0, B_i = 1
    P10 = []  # Positions where A_i = 1, B_i = 0
    
    for i in range(N):
        if A[i] == 0 and B[i] == 1:
            P01.append(i)
        elif A[i] == 1 and B[i] == 0:
            P10.append(i)
    
    # Sort P10 in decreasing order of C_i
    P10.sort(key=lambda i: C[i], reverse=True)
    
    # Sort P01 in increasing order of C_i
    P01.sort(key=lambda i: C[i])
    
    # Flip positions in P10 first and then in P01
    order = P10 + P01
    
    total_cost = 0
    current_cost = sum(A[j] * C[j] for j in range(N))
    
    for i in order:
        cost_change = (1 - 2 * A[i]) * C[i]
        A[i] = 1 - A[i]  # Flip the bit
        current_cost += cost_change
        total_cost += current_cost
    
    return total_cost

def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    result = min_cost_to_match(A, B, C)
    print(result)

if __name__ == "__main__":
    main()