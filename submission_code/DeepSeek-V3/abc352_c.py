def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    B = []
    index = 1
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index+1]))
        index += 2
    
    # Calculate the difference B_i - A_i
    diff = [B[i] - A[i] for i in range(N)]
    
    # Sort the giants based on the difference in descending order
    # We want to maximize the sum of A_i and the sum of (B_i - A_i) for the last giant
    # So, we sort the giants such that the one with the largest (B_i - A_i) is last
    # To achieve this, we sort the giants in ascending order of (B_i - A_i)
    # Because when we stack them, the last one will have the largest (B_i - A_i)
    
    # Create a list of tuples (A_i, B_i, diff_i)
    giants = list(zip(A, B, diff))
    
    # Sort the giants based on the difference in ascending order
    giants.sort(key=lambda x: x[2])
    
    # Now, stack the giants in the order of the sorted list
    total_height = 0
    max_head_height = 0
    for i in range(N):
        a, b, d = giants[i]
        total_height += a
        if i == N-1:
            max_head_height = total_height + d
        else:
            pass
    
    print(max_head_height)

if __name__ == "__main__":
    main()