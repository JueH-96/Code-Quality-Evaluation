# YOUR CODE HERE
def maximize_string_value():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    S = list(data[2])
    T = list(data[3])
    
    # Sort T in descending order to use the largest digits first
    T.sort(reverse=True)
    
    # Pointer to track the position in T
    t_index = 0
    
    # Iterate over S and replace with T's largest available digits
    for i in range(N):
        if t_index < M and S[i] < T[t_index]:
            S[i] = T[t_index]
            t_index += 1
        if t_index >= M:
            break
    
    # Join the list back to a string and print the result
    print(''.join(S))