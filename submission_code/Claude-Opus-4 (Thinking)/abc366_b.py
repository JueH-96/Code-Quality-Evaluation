# YOUR CODE HERE
n = int(input())
strings = []
for _ in range(n):
    strings.append(input())

m = max(len(s) for s in strings)

# Create T_1 to T_m
for j in range(m):
    t_j = ['*'] * n
    
    for i in range(n):
        if j < len(strings[i]):
            t_j[n-i-1] = strings[i][j]
    
    # Convert to string and remove trailing '*'
    t_j_str = ''.join(t_j)
    t_j_str = t_j_str.rstrip('*')
    
    print(t_j_str)