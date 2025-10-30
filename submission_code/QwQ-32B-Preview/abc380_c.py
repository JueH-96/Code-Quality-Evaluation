def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Find all 1-blocks
    blocks = []
    start = None
    for i in range(N):
        if S[i] == '1':
            if start is None:
                start = i
            if i == N-1 or S[i+1] == '0':
                end = i
                blocks.append((start, end))
                start = None
        else:
            if start is not None:
                # '1's are continuous, do nothing
                pass
            else:
                # '0', do nothing
                pass
    
    # Ensure there are at least K blocks
    if len(blocks) < K:
        print("Error: Not enough 1-blocks in the string.")
        return
    
    # Get the (K-1)-th and K-th blocks
    end_k_minus_1 = blocks[K-2][1]
    start_k = blocks[K-1][0]
    end_k = blocks[K-1][1]
    
    # Construct the new string
    part1 = S[0:end_k_minus_1 + 1]
    part2 = S[start_k:end_k + 1]
    part3 = S[end_k_minus_1 + 1:start_k]
    part4 = S[end_k + 1:]
    result = part1 + part2 + part3 + part4
    
    # Print the result
    print(result)

if __name__ == "__main__":
    main()