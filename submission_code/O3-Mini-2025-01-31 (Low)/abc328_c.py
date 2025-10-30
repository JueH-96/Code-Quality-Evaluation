def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    Q = int(input_data[1])
    S = input_data[2]
    
    # Create prefix sum array: prefix[i] number of consecutive same pairs in S[0:i]
    prefix = [0] * (N + 1)
    for i in range(1, N):
        prefix[i + 1] = prefix[i] + (1 if S[i] == S[i - 1] else 0)
    
    # Process queries
    result = []
    index = 3
    for _ in range(Q):
        l = int(input_data[index])
        r = int(input_data[index+1])
        index += 2
        # Direct answer: prefix[r] - prefix[l]
        result.append(str(prefix[r] - prefix[l]))
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()