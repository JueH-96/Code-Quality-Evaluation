import bisect
import math
from sys import stdin

def main():
    input = stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    H = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Compute next_greater for 1-based indices
    next_greater = [-1] * (N + 2)  # next_greater[1..N]
    stack = []
    for i in range(N, 0, -1):
        current_h = H[i-1]
        while stack and H[stack[-1]-1] <= current_h:
            stack.pop()
        if stack:
            next_greater[i] = stack[-1]
        else:
            next_greater[i] = -1
        stack.append(i)
    
    # Build S_r for each r (1-based)
    S = [[] for _ in range(N+2)]  # S[r] is the list for r
    for r in range(1, N+1):
        current = next_greater[r]
        s_list = []
        while current != -1:
            s_list.append(H[current-1])
            current = next_greater[current]
        S[r] = s_list
    
    # Build sparse table for range maximum query
    log_n = math.floor(math.log2(N)) + 1 if N > 0 else 0
    st = []
    st.append(H.copy())
    for k in range(1, log_n):
        prev = st[k-1]
        curr = [0] * N
        length = 1 << (k-1)
        for i in range(N - (1 << k) + 1):
            curr[i] = max(prev[i], prev[i + length])
        st.append(curr)
    
    def get_max(l, r):
        if l > r:
            return 0
        length = r - l + 1
        k = length.bit_length() - 1
        max_val = max(st[k][l], st[k][r - (1 << k) + 1])
        return max_val
    
    for _ in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        
        # Compute M
        l_query = l + 1
        r_query = r - 1
        if l_query > r_query:
            M = 0
        else:
            # Convert to 0-based
            l_0 = l_query - 1
            r_0 = r_query - 1
            if l_0 < 0 or r_0 >= N:
                M = 0
            else:
                M = get_max(l_0, r_0)
        
        # Get S[r]
        s_list = S[r]
        # Find the first index >= M
        count = len(s_list) - bisect.bisect_left(s_list, M)
        print(count)

if __name__ == '__main__':
    main()