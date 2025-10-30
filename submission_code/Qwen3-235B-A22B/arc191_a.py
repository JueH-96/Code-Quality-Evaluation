import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    T = data[3]
    
    # Sort T in descending order
    sorted_T = sorted(T, reverse=True)
    
    # Use a min-heap like approach to replace leftmost possible characters
    j = 0
    for t_char in sorted_T:
        # Find the leftmost character in S that can be replaced
        while j < N and S[j] >= t_char:
            j += 1
        if j < N:
            S[j] = t_char
            j += 1
    
    print(''.join(S))

if __name__ == '__main__':
    main()