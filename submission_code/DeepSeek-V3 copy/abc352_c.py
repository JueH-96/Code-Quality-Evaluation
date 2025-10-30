def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    B = []
    for i in range(N):
        A.append(int(data[1 + 2*i]))
        B.append(int(data[2 + 2*i]))
    
    # Calculate the difference B_i - A_i for each giant
    diff = [B[i] - A[i] for i in range(N)]
    
    # Sort the giants based on the difference in descending order
    # We want to maximize the sum of A_i and the last B_i - A_i
    # So, we sort the giants such that the one with the largest B_i - A_i is at the top
    # But since the sum of A_i is fixed, the order of A_i does not matter
    # So, we can arrange the giants in any order, but the last one should have the largest B_i - A_i
    
    # To maximize the head height of the topmost giant, we need to maximize the sum of A_i for all giants except the last one, plus the B_i of the last one
    # Since the sum of A_i is fixed, the only way to maximize the head height is to have the last giant with the largest B_i - A_i
    
    # So, we sort the giants based on B_i - A_i in ascending order, and place the one with the largest B_i - A_i at the end
    
    # Create a list of tuples (A_i, B_i, diff_i)
    giants = list(zip(A, B, diff))
    
    # Sort the giants based on diff in ascending order
    giants.sort(key=lambda x: x[2])
    
    # Now, arrange the giants in the order of the sorted list, but place the last one with the largest diff
    # So, the order is giants[0], giants[1], ..., giants[N-2], giants[N-1]
    
    # Calculate the total height
    total_height = 0
    for i in range(N-1):
        total_height += giants[i][0]
    total_height += giants[N-1][1]
    
    print(total_height)

if __name__ == "__main__":
    main()