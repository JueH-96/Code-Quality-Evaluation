# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # First line: N, M, K
    N, M, K = map(int, data[0].split())
    
    # Following lines: C_i, A_{i,1}, ..., A_{i,C_i}, R_i
    tests = []
    for i in range(1, M + 1):
        line = data[i].split()
        C_i = int(line[0])
        A_i = list(map(int, line[1:C_i + 1]))
        R_i = line[C_i + 1]
        tests.append((C_i, A_i, R_i))
    
    # Total number of combinations of real/dummy keys
    total_combinations = 1 << N  # 2^N
    
    valid_combinations = 0
    
    # Iterate over all possible combinations of real/dummy keys
    for mask in range(total_combinations):
        is_valid = True
        
        for C_i, A_i, R_i in tests:
            real_count = sum((mask >> (key - 1)) & 1 for key in A_i)
            
            if R_i == 'o':
                if real_count < K:
                    is_valid = False
                    break
            elif R_i == 'x':
                if real_count >= K:
                    is_valid = False
                    break
        
        if is_valid:
            valid_combinations += 1
    
    print(valid_combinations)

if __name__ == "__main__":
    main()