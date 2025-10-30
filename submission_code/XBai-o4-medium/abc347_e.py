import sys

def main():
    n, q = map(int, sys.stdin.readline().split())
    x_list = list(map(int, sys.stdin.readline().split()))
    
    # Initialize toggles for each element
    toggles = [[] for _ in range(n + 1)]  # 1-based indexing
    for i in range(q):
        x = x_list[i]
        toggles[x].append(i)
    
    # Compute the c array (size of S after each query)
    present = [False] * (n + 1)
    current_size = 0
    c = [0] * q
    for i in range(q):
        x = x_list[i]
        if present[x]:
            current_size -= 1
            present[x] = False
        else:
            current_size += 1
            present[x] = True
        c[i] = current_size
    
    # Compute prefix sums of the c array
    prefix = [0] * (q + 1)
    for i in range(q):
        prefix[i + 1] = prefix[i] + c[i]
    
    # Calculate the final A array
    A = [0] * (n + 1)  # A[1..n]
    for j in range(1, n + 1):
        tls = toggles[j]
        m = len(tls)
        total = 0
        for k in range(0, m, 2):
            start = tls[k]
            if k + 1 < m:
                end = tls[k + 1]
            else:
                end = q
            total += prefix[end] - prefix[start]
        A[j] = total
    
    # Output the result
    print(' '.join(map(str, A[1:n + 1])))

if __name__ == '__main__':
    main()