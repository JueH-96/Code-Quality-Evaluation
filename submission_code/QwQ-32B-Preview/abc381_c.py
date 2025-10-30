def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    # Find all positions of '/'
    slash_positions = [i for i in range(N) if S[i] == '/']
    
    # Compute left_ones
    left_ones = [0] * N
    for i in range(1, N):
        if S[i-1] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0
    
    # Compute right_twos
    right_twos = [0] * N
    for i in range(N-2, -1, -1):
        if S[i+1] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
    
    max_length = 0
    for i in slash_positions:
        k = min(left_ones[i], right_twos[i])
        length = 2 * k + 1
        if length > max_length:
            max_length = length
    
    print(max_length)

if __name__ == "__main__":
    main()