def solve(S, X, Y):
    count_S_X = X.count('0')
    count_T_X = X.count('1')
    count_S_Y = Y.count('0')
    count_T_Y = Y.count('1')
    
    a = count_S_X - count_S_Y
    b = count_T_Y - count_T_X
    
    if b == 0:
        if a == 0:
            return "Yes"
        else:
            return "No"
    else:
        if a == 0:
            # T must be empty
            seq_X = S * count_S_X
            seq_Y = S * count_S_Y
            if seq_X == seq_Y:
                return "Yes"
            else:
                return "No"
        else:
            # Check if len(S) * a is divisible by b
            total_len_S = len(S) * a
            if total_len_S % b != 0:
                return "No"
            else:
                len_T = total_len_S // b
                # Compute T
                T = S * a[:len_T]
                # Check if T * b == a * S
                if T * b == S * a:
                    return "Yes"
                else:
                    return "No"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    t = int(data[0])
    index = 1
    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        print(solve(S, X, Y))
        index += 3

if __name__ == "__main__":
    main()