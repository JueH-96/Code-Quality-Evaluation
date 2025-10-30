def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    A = list(map(int, input[1:n+1]))
    stack = []
    res = [[0, 0] for _ in range(n)]
    for i in range(n):
        current_sum = A[i]
        current_max = A[i]
        while stack and stack[-1][1] < current_sum:
            prev_sum, prev_max = stack.pop()
            current_sum += prev_sum
            current_max = max(current_max, prev_max)
        stack.append((current_sum, current_max))
        res[i] = (current_sum, current_max)
    
    max_sum_list = [0]*n
    for i in range(n):
        if i ==0 or res[i][0] > max_sum_list[i-1]:
            max_sum_list[i] = res[i][0]
        else:
            max_sum_list[i] = max_sum_list[i-1]
    
    stack = []
    right_max_list = [0]*n
    for i in range(n-1, -1, -1):
        current_sum = A[i]
        current_max = A[i]
        while stack and stack[-1][1] < current_sum:
            prev_sum, prev_max = stack.pop()
            current_sum += prev_sum
            current_max = max(current_max, prev_max)
        stack.append((current_sum, current_max))
        right_max_list[i] = current_sum
        
    answer = []
    for i in range(n):
        answer.append(str(max(res[i][0], right_max_list[i])))
    print(' '.join(answer))

if __name__ == "__main__":
    main()