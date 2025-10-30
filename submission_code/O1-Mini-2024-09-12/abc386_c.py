# YOUR CODE HERE
import sys

def can_convert_with_one_operation(K, S, T):
    if S == T:
        return True
    if K <1:
        return False
    len_s = len(S)
    len_t = len(T)
    if abs(len_s - len_t) >1:
        return False
    if len_s == len_t:
        diff = 0
        for a, b in zip(S, T):
            if a != b:
                diff +=1
                if diff >1:
                    return False
        return diff <=1
    # Ensure S is shorter or equal for insertion, or longer or equal for deletion
    if len_s +1 == len_t:
        # Check if S can be obtained by deleting one character from T
        i = j =0
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i +=1
                j +=1
            else:
                j +=1
                if j -i >1:
                    return False
        return True
    if len_s -1 == len_t:
        # Check if T can be obtained by deleting one character from S
        i = j =0
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i +=1
                j +=1
            else:
                i +=1
                if i -j >1:
                    return False
        return True
    return False

def main():
    input = sys.stdin.read().split()
    K = int(input[0])
    S = input[1]
    T = input[2]
    if can_convert_with_one_operation(K, S, T):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()