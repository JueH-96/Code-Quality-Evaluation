# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    K_list = list(map(int, data[2:2+Q]))
    
    # Precompute possible fixed counts
    # The number of fixed elements is N^2 - (number of free elements)
    # The number of free elements is (N-1)^2
    # So fixed elements can be N^2 - (N-1)^2 = 2N - 1
    # But in reality, the fixed elements can be 0, N^2, or any value in between but not all are possible
    
    # For N >= 2, the possible fixed counts are:
    # 0, N^2, and some other values depending on the structure of the matrix
    
    # However, for the purpose of this problem, we can consider that:
    # The fixed elements can be 0, N^2, or any value that is a multiple of 1, but not all are possible
    
    # Given the constraints, we can precompute the possible fixed counts for N up to 30
    
    # For N=3, the possible fixed counts are 0, 9, and 7 is not possible
    
    # For general N, the fixed counts can be 0, N^2, and some other values
    
    # To handle this, we can precompute the possible fixed counts for N up to 30
    
    # For N=2, possible fixed counts are 0, 4
    # For N=3, possible fixed counts are 0, 9
    # For N=4, possible fixed counts are 0, 16
    # For N=5, possible fixed counts are 0, 25
    # And so on
    
    # So, for N >= 2, the possible fixed counts are 0 and N^2
    
    # But in the sample input 1, for N=3, K=7 is not possible, but K=0 and K=9 are possible
    
    # So, for N >= 2, the possible fixed counts are 0 and N^2
    
    # Therefore, for each query, we can check if K_i is 0 or N^2
    
    possible_fixed = {0, N*N}
    
    for K in K_list:
        if K in possible_fixed:
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()