def find_top_two(lst):
    max_val = -1
    second_max = -1
    for val in lst:
        if val > max_val:
            second_max = max_val
            max_val = val
        elif val > second_max:
            second_max = val
    return max_val, second_max

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    A = []
    B = []
    
    index = 2
    for _ in range(N):
        A.append(int(data[index]))
        B.append(int(data[index + 1]))
        index += 2
    
    dp_prev = [A[i] + B[i] for i in range(N)]
    
    for k in range(2, K + 1):
        max_prev, second_max_prev = find_top_two(dp_prev)
        dp_prev = [A[i] * second_max_prev + B[i] if dp_prev[i] == max_prev else A[i] * max_prev + B[i] for i in range(N)]
    
    print(max(dp_prev))

if __name__ == "__main__":
    main()