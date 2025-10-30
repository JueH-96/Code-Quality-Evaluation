# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    C = list(map(int, data[3:]))
    
    from collections import defaultdict
    
    color_indices = defaultdict(list)
    
    for i in range(N):
        color_indices[C[i]].append(i)
    
    for i in range(1, M + 1):
        indices = color_indices[i]
        if len(indices) > 1:
            last_char = S[indices[-1]]
            for j in range(len(indices) - 1, 0, -1):
                S[indices[j]] = S[indices[j - 1]]
            S[indices[0]] = last_char
    
    print("".join(S))

if __name__ == "__main__":
    main()