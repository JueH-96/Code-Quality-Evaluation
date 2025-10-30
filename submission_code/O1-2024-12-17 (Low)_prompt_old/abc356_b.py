def solve():
    import sys
    data = sys.stdin.read().strip().split()
    
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    offset = 2+M
    
    # Initialize sums for each nutrient to 0
    nutrient_sums = [0]*M
    
    # Read the matrix of X_{i,j}
    for i in range(N):
        row = list(map(int, data[offset:offset+M]))
        offset += M
        for j in range(M):
            nutrient_sums[j] += row[j]
    
    # Check if goals are met
    for j in range(M):
        if nutrient_sums[j] < A[j]:
            print("No")
            return
    
    print("Yes")
    
def main():
    solve()

if __name__ == "__main__":
    main()