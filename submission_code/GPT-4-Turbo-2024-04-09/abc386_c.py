# YOUR CODE HERE
import sys
input = sys.stdin.read

def can_transform_with_one_operation(S, T):
    len_s = len(S)
    len_t = len(T)
    
    # If the lengths differ by more than 1, it's impossible with one operation
    if abs(len_s - len_t) > 1:
        return False
    
    # If lengths are the same, we can check for replacement
    if len_s == len_t:
        differences = sum(1 for i in range(len_s) if S[i] != T[i])
        return differences <= 1
    
    # If lengths differ by exactly one, check for insertion or deletion
    if len_s < len_t:
        # S needs an insertion to become T
        # Try to find a position where characters before are the same and after insertion they remain the same
        for i in range(len_s + 1):
            if S[:i] + T[i] + S[i:] == T:
                return True
        return False
    else:
        # S needs a deletion to become T
        # Try to find a position to delete one character from S to make it T
        for i in range(len_s):
            if S[:i] + S[i+1:] == T:
                return True
        return False

def main():
    data = input().split()
    K = int(data[0])
    S = data[1]
    T = data[2]
    
    if can_transform_with_one_operation(S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()