import sys
from collections import deque

def main():
    K = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    len_S = len(S)
    len_T = len(T)
    
    # Initialize visited sets for each possible number of operations
    visited = [set() for _ in range(K + 2)]
    queue = deque()
    visited[0].add((0, 0))
    queue.append((0, 0, 0))
    
    while queue:
        i, j, ops = queue.popleft()
        
        # Check if we have reached the end of both strings
        if i == len_S and j == len_T:
            print("Yes")
            return
        
        if ops > K:
            continue
        
        # Insert operation: add a character to match T[j], so j increases by 1
        new_i_ins = i
        new_j_ins = j + 1
        new_ops_ins = ops + 1
        if new_ops_ins <= K:
            if (new_i_ins, new_j_ins) not in visited[new_ops_ins]:
                visited[new_ops_ins].add((new_i_ins, new_j_ins))
                queue.append((new_i_ins, new_j_ins, new_ops_ins))
        
        # Delete operation: delete a character from S, so i increases by 1
        new_i_del = i + 1
        new_j_del = j
        new_ops_del = ops + 1
        if new_ops_del <= K:
            if (new_i_del, new_j_del) not in visited[new_ops_del]:
                visited[new_ops_del].add((new_i_del, new_j_del))
                queue.append((new_i_del, new_j_del, new_ops_del))
        
        # Replace operation: check if current characters match
        if i < len_S and j < len_T:
            if S[i] == T[j]:
                new_ops_rep = ops
            else:
                new_ops_rep = ops + 1
            new_i_rep = i + 1
            new_j_rep = j + 1
            if new_ops_rep <= K:
                if (new_i_rep, new_j_rep) not in visited[new_ops_rep]:
                    visited[new_ops_rep].add((new_i_rep, new_j_rep))
                    queue.append((new_i_rep, new_j_rep, new_ops_rep))
    
    # If we exhaust all possibilities without finding a match
    print("No")

if __name__ == "__main__":
    main()