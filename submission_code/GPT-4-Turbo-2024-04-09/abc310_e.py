def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # NAND operation function
    def nand(x, y):
        return 1 if x == 0 or y == 0 else 0
    
    total_sum = 0
    
    # Iterate over all possible starting points i
    for i in range(N):
        current_value = int(S[i])
        # Iterate over all possible ending points j >= i
        for j in range(i, N):
            current_value = nand(current_value, int(S[j]))
            total_sum += current_value
    
    print(total_sum)