# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:N+1]
    
    M = max(len(s) for s in S)
    
    # Initialize T as a list of empty strings
    T = ['' for _ in range(M)]
    
    for i in range(N):
        s = S[i]
        len_s = len(s)
        for j in range(M):
            if j < len_s:
                # The (N-i)th character in T[j] should be s[j]
                # Since T is built from the end, we need to adjust the index
                # For the first string (i=0), it should be the last in T
                # So, for T[j], the (N-i-1)th character is s[j]
                # Wait, perhaps it's easier to build T in reverse order
                # Let's think differently: for each T[j], the (N-i)th character is s[j]
                # So, for T[j], we need to pad with '*' until the (N-i)th position
                # Maybe it's easier to build T[j] as a list and then join
                # Initialize T[j] as a list of '*' of length N
                if len(T[j]) < N:
                    T[j] += '*' * (N - len(T[j]))
                # Now, replace the (N-i-1)th character with s[j]
                T[j] = T[j][:N-i-1] + s[j] + T[j][N-i:]
            else:
                # For j >= len_s, the (N-i)th character should be '*' or not exist
                # Since T[j] is already initialized with '*' up to N characters, no need to do anything
                pass
    
    # Now, ensure that each T[j] does not end with '*'
    for j in range(M):
        T[j] = T[j].rstrip('*')
    
    # Print the T strings
    for t in T:
        print(t)

if __name__ == "__main__":
    main()