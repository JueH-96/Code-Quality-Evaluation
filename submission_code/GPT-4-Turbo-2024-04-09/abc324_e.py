def is_subsequence(s, t):
    it = iter(s)
    return all(c in it for c in t)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = data[1]
    strings = data[2:]
    
    count = 0
    
    # Check each pair (i, j)
    for i in range(N):
        for j in range(N):
            # Concatenate S_i and S_j
            combined = strings[i] + strings[j]
            # Check if T is a subsequence of the combined string
            if is_subsequence(combined, T):
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()