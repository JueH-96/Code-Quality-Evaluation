import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    x_list = list(map(int, input[ptr:ptr+Q]))
    
    # Compute the size of S after each query
    present = [False] * (N + 1)
    s_arr = []
    s = 0
    for x in x_list:
        if present[x]:
            present[x] = False
            s -= 1
        else:
            present[x] = True
            s += 1
        s_arr.append(s)
    
    # Compute prefix sums of s_arr
    prefix_sum = [0] * (len(s_arr) + 1)
    for i in range(len(s_arr)):
        prefix_sum[i + 1] = prefix_sum[i] + s_arr[i]
    
    # Track intervals when each element is present in S
    current_in = [False] * (N + 1)
    current_start = [-1] * (N + 1)
    intervals = [[] for _ in range(N + 1)]
    for idx in range(len(x_list)):
        x = x_list[idx]
        if current_in[x]:
            # Remove x from S
            current_in[x] = False
            start = current_start[x]
            end = idx - 1
            intervals[x].append((start, end))
            current_start[x] = -1
        else:
            # Add x to S
            current_in[x] = True
            current_start[x] = idx
    
    # Add intervals for elements still in S after all queries
    for j in range(1, N + 1):
        if current_in[j]:
            start = current_start[j]
            end = len(x_list) - 1
            intervals[j].append((start, end))
    
    # Calculate the result for each element
    A = [0] * (N + 1)
    for j in range(1, N + 1):
        for (start, end) in intervals[j]:
            A[j] += prefix_sum[end + 1] - prefix_sum[start]
    
    # Output the result
    print(' '.join(map(str, A[1:N+1])))

if __name__ == "__main__":
    main()