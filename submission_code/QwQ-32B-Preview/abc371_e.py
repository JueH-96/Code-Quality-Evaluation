def main():
    import sys
    tokens = sys.stdin.read().split()
    N = int(tokens[0])
    A = list(map(int, tokens[1:N+1]))
    
    pos_dict = {}
    for idx, val in enumerate(A):
        if val not in pos_dict:
            pos_dict[val] = []
        pos_dict[val].append(idx)
    
    total_subarrays = N * (N + 1) // 2
    total_sum = 0
    
    for x in pos_dict.values():
        subarrays_without_x = 0
        if x[0] > 0:
            length = x[0]
            subarrays_without_x += length * (length + 1) // 2
        for i in range(1, len(x)):
            if x[i] - x[i-1] > 1:
                length = x[i] - x[i-1] - 1
                subarrays_without_x += length * (length + 1) // 2
        if x[-1] < N - 1:
            length = N - 1 - x[-1]
            subarrays_without_x += length * (length + 1) // 2
        number_of_subarrays_with_x = total_subarrays - subarrays_without_x
        total_sum += number_of_subarrays_with_x
    
    print(total_sum)

if __name__ == "__main__":
    main()