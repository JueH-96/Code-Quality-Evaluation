# YOUR CODE HERE
import sys
import sys
import sys

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1:N+1]
    
    INF = float('inf')
    children = [[-1]*26]
    min_length = [INF]
    node_count = 1  # Next node index to assign
    
    for idx in range(N):
        s = S[idx]
        l_s = len(s)
        if idx ==0:
            min_cost = l_s
        else:
            current = 0
            min_cost = l_s  # Option to make T empty
            min_x_minus_2l = INF
            for l, c in enumerate(s,1):
                c_idx = ord(c) - ord('a')
                if children[current][c_idx] == -1:
                    break
                current = children[current][c_idx]
                x_len = min_length[current]
                if x_len != INF:
                    cost = l_s + x_len - 2*l
                    if cost < min_cost:
                        min_cost = cost
            # Compare to making T empty is already considered
        print(min_cost)
        
        # Insert S_k into trie
        current =0
        for c in s:
            c_idx = ord(c) - ord('a')
            if children[current][c_idx] == -1:
                children.append([-1]*26)
                min_length.append(INF)
                children[current][c_idx] = node_count
                node_count +=1
            current = children[current][c_idx]
            if min_length[current] > l_s:
                min_length[current] = l_s

if __name__ == "__main__":
    main()