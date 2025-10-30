def hamming_distance(N, S, T):
    # Initialize the distance counter
    distance = 0
    
    # Iterate through each character in the strings
    for i in range(N):
        if S[i] != T[i]:
            distance += 1
            
    return distance

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    S = data[1]
    T = data[2]
    
    result = hamming_distance(N, S, T)
    print(result)