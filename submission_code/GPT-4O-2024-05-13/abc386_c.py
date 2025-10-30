# YOUR CODE HERE
import sys

def can_convert_with_one_operation(K, S, T):
    if K != 1:
        return "No"
    
    len_s = len(S)
    len_t = len(T)
    
    if S == T:
        return "Yes"
    
    if abs(len_s - len_t) > 1:
        return "No"
    
    if len_s == len_t:
        # Check for one replacement
        diff_count = sum(1 for a, b in zip(S, T) if a != b)
        if diff_count == 1:
            return "Yes"
    elif len_s + 1 == len_t:
        # Check for one insertion
        for i in range(len_t):
            if S == T[:i] + T[i+1:]:
                return "Yes"
    elif len_s == len_t + 1:
        # Check for one deletion
        for i in range(len_s):
            if T == S[:i] + S[i+1:]:
                return "Yes"
    
    return "No"

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    K = int(data[0])
    S = data[1]
    T = data[2]
    print(can_convert_with_one_operation(K, S, T))