# YOUR CODE HERE
import sys
import sys
def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr]); ptr +=1
    S = list(data[ptr]); ptr +=1
    Q = int(data[ptr]); ptr +=1
    last_individual_time = [0]*N
    last_global_time = 0
    last_global_op = None
    for t in range(1,Q+1):
        t_i = int(data[ptr]); ptr +=1
        x_i = int(data[ptr]); ptr +=1
        c_i = data[ptr]; ptr +=1
        if t_i ==1:
            idx = x_i -1
            S[idx] = c_i
            last_individual_time[idx] = t
        elif t_i ==2 or t_i ==3:
            last_global_time = t
            last_global_op = t_i
    if last_global_op:
        for i in range(N):
            if last_individual_time[i] <= last_global_time:
                if last_global_op ==2:
                    if S[i].isupper():
                        S[i] = S[i].lower()
                elif last_global_op ==3:
                    if S[i].islower():
                        S[i] = S[i].upper()
    print(''.join(S))
    
if __name__ == "__main__":
    main()