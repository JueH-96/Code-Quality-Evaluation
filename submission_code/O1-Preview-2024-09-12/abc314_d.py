# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    Q = int(sys.stdin.readline())
    S_list = list(S)
    letter_last_change = [0]*N
    last_case_change_type = None  # None, 2, or 3
    last_case_change_index = 0
    for idx in range(1, Q+1):
        parts = sys.stdin.readline().split()
        t = int(parts[0])
        if t ==1:
            x_i = int(parts[1])
            c_i = parts[2]
            pos = x_i -1
            S_list[pos] = c_i
            letter_last_change[pos] = idx
        else:
            # t ==2 or t ==3
            last_case_change_type = t
            last_case_change_index = idx
    # Output the final string
    result = []
    for i in range(N):
        if letter_last_change[i] > last_case_change_index:
            # The character was modified after the last case change operation
            result.append(S_list[i])
        else:
            # Apply the case change if any
            if last_case_change_type == 2:
                result.append(S_list[i].lower())
            elif last_case_change_type == 3:
                result.append(S_list[i].upper())
            else:
                result.append(S_list[i])
    print(''.join(result))
    
threading.Thread(target=main).start()