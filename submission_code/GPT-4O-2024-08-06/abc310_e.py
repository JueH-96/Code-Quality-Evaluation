# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1]
    
    A = [int(c) for c in S]
    
    total_sum = 0
    
    for i in range(N):
        current_nand = A[i]
        total_sum += current_nand
        for j in range(i + 1, N):
            current_nand = 1 - (current_nand & A[j])  # NAND operation
            total_sum += current_nand
    
    print(total_sum)

if __name__ == "__main__":
    main()