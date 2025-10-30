import sys
from collections import defaultdict

def main():
    n, q = map(int, sys.stdin.readline().split())
    x_list = list(map(int, sys.stdin.readline().split()))
    
    toggles = defaultdict(list)
    s = set()
    size_t = [0] * (q + 1)
    
    for step in range(1, q + 1):
        x_i = x_list[step - 1]
        if x_i in s:
            s.remove(x_i)
            size_t[step] = size_t[step - 1] - 1
        else:
            s.add(x_i)
            size_t[step] = size_t[step - 1] + 1
        toggles[x_i].append(step)
    
    # Compute prefix sums
    prefix = [0] * (q + 1)
    for step in range(1, q + 1):
        prefix[step] = prefix[step - 1] + size_t[step]
    
    A = [0] * (n + 1)  # 1-based indexing
    
    for j in range(1, n + 1):
        lst = toggles[j]
        m = len(lst)
        total = 0
        for i in range(m):
            if i % 2 == 0:  # Add event
                start = lst[i]
                if (i + 1) < m:
                    end = lst[i + 1]
                    # Sum from start to end - 1 inclusive
                    total += prefix[end - 1] - prefix[start - 1]
                else:
                    # Sum from start to q inclusive
                    total += prefix[q] - prefix[start - 1]
        A[j] = total
    
    # Output A[1..n]
    print(' '.join(map(str, A[1:n+1])))

if __name__ == "__main__":
    main()