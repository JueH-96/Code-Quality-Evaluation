def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    a = list(map(int, data[3:3+N]))
    b = list(map(int, data[3+N:3+N+M]))
    
    forbidden = set()
    for i in range(3+N+M, 3+N+M+2*L, 2):
        main = int(data[i]) - 1  # 0-indexed
        side = int(data[i+1]) - 1  # 0-indexed
        forbidden.add((main, side))
    
    # Sort main and side dishes in descending order, keep original indices
    sorted_a = sorted(enumerate(a), key=lambda x: -x[1])
    sorted_b = sorted(enumerate(b), key=lambda x: -x[1])
    
    i = 0
    j = 0
    max_sum = 0
    
    while i < N and j < M:
        original_main_index = sorted_a[i][0]
        original_side_index = sorted_b[j][0]
        
        if (original_main_index, original_side_index) not in forbidden:
            current_sum = sorted_a[i][1] + sorted_b[j][1]
            if current_sum > max_sum:
                max_sum = current_sum
            # Try to increase j to check for higher sums with the same main dish
            if j + 1 < M and sorted_a[i][1] + sorted_b[j+1][1] > max_sum:
                j += 1
            else:
                i += 1
        else:
            # If the pair is forbidden, move to the next best option
            if i + 1 < N and a[sorted_a[i+1][0]] + b[sorted_b[j][0]] > a[sorted_a[i][0]] + b[sorted_b[j+1][0]]:
                i += 1
            else:
                j += 1
    
    print(max_sum)

if __name__ == '__main__':
    main()